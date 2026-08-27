"""
DC power flow (DCPF) plugin for ZEN-garden.

Adds Kirchhoff's voltage law (KVL) to a user-selected part of the network, so a
single model can combine a physically modelled region (e.g. a detailed German
grid) with a transport-model representation everywhere else.

Formulation
-----------
ZEN-garden represents every physical line as two directed edges with
non-negative flows, so KVL can be written in two ways. The plugin supports both
and picks between them automatically, because neither is correct in all cases.

**signed** — the reverse edge is pinned to zero and the forward edge is allowed
to go negative::

    F_{j,e+,t} = B_e (theta_{u,t} - theta_{v,t}),   F_{j,e-,t} = 0

Flows are unique and directly interpretable. This is only valid on a *lossless,
cost-free* line: ``flow_transport_loss`` and ``cost_opex_variable`` are both
bounded below by zero and tied by equality to ``factor * flow_transport``, so a
negative flow makes the model infeasible the moment a loss factor, a variable
opex or a carbon intensity is non-zero. The same holds for on-off behaviour: a
non-zero ``min_load`` puts the technology in ``set_on_off``, whose constraints
force ``flow_transport >= 0``. The plugin verifies all three parameters are
zero and that no on-off constraints exist on the KVL edges, and refuses to use
this mode otherwise.

**net** — both directed flows stay non-negative and KVL is imposed on their
difference::

    F_{j,e+,t} - F_{j,e-,t} = B_e (theta_{u,t} - theta_{v,t})

Always valid: losses, opex, emissions and both capacity constraints keep their
stock semantics. But when losses and opex are zero the pair is determined only
up to a common additive constant, so the solver may return large flows in both
directions whose difference is correct. Dispatch, angles and ``|net| <= capacity``
remain right, but per-direction flows become uninterpretable, per-direction
capacity constraints can sit tight for purely degenerate reasons, and the
resulting primal degeneracy makes dual values non-unique — which matters if
nodal prices are an output. Call :func:`report_circulation` to quantify it, and
net the directed flows before reporting anything derived from them.

``flow_representation`` selects the mode: ``"auto"`` (default) uses **signed**
where the zero-loss, zero-cost check passes and **net** elsewhere, logging which
and why; ``"signed"`` and ``"net"`` force one, the former raising if the check
fails. A lossless DC power flow model — the usual case — therefore gets unique,
interpretable flows without any configuration.

Scope selection
---------------
Impedance cannot double as the on/off switch: ``impedance`` is declared in
``attributes.json`` with a ``default_value``, so an edge missing from
``impedance.csv`` silently receives a valid impedance rather than a marker.
Membership is therefore declared as a *node set*, which is also the natural way
to express "detailed DE, transport elsewhere". An edge is KVL-constrained when
**both** of its endpoints are in that set; boundary edges leaving the region
stay transport edges, which is the intended behaviour.

Slack buses
-----------
KVL fixes voltage angles only up to a constant per synchronous island, so one
reference bus is needed per *connected component* of the KVL subgraph, not one
globally. The components are computed from the selected lines and a slack is
pinned in each.

Investable corridors
--------------------
KVL is a hard equality. On a corridor the optimizer may leave unbuilt
(``capacity_existing`` zero, additions allowed), a zero capacity pins the flow
to zero and KVL degenerates to ``theta_u == theta_v`` — the unbuilt line acts
as a zero-impedance short instead of an open circuit, which can force
investment the network does not need. Modelling that correctly needs a
disjunctive (big-M) formulation keyed to the build decision, which this plugin
does not implement. ``investment_mode`` controls how such corridors are
handled: ``"warn"`` (default) logs them and proceeds, ``"error"`` aborts
construction so the hazard cannot pass silently.

Configuration
-------------
Declared under ``plugins`` in the run config::

    "plugins": {
      "dclf": {
        "kvl_nodes":           ["DE*"],   // names or fnmatch patterns; default all
        "technologies":        ["power_lines"], // default: techs with impedance
        "impedance_file":      "impedance",
        "flow_representation": "auto",    // "auto" | "signed" | "net"
        "investment_mode":     "warn",    // "warn" | "error" — unbuilt KVL corridors
        "slack_nodes":         [],        // optional; else one chosen per component
        "verbose":             false
      }
    }

Setting ``kvl_nodes`` to ``[]`` or omitting it applies KVL to the whole network,
which reproduces a conventional DC OPF.
"""

from __future__ import annotations

import fnmatch
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from zen_garden.events import Event, Events
from zen_garden.model.technology.transport_technology import TransportTechnology
from zen_garden.plugins.network_utils import (
    build_reverse_edge_map,
    describe_ambiguous_edges,
    pair_key,
)

# Populated by loader.py from the run config
config = {}

# Toggled by loader.py; the hook is process-global once imported, so runs that
# do not select this plugin must not execute it.
enabled = False

LOG_PREFIX = "DCPF plugin:"


class DCPFDataError(ValueError):
    """Raised when the network data cannot support the requested DCPF scope."""


# --------------------------------------------------------------------------- #
# Scope resolution
# --------------------------------------------------------------------------- #


def _resolve_kvl_nodes(patterns, nodes):
    """Expand node name patterns into an explicit node set.

    :param patterns: list of node names or fnmatch patterns; empty means all
    :param nodes: all nodes in the system
    :return: sorted list of selected node names
    """
    if not patterns:
        return sorted(nodes)
    selected = {n for n in nodes for p in patterns if fnmatch.fnmatch(n, p)}
    unmatched = [p for p in patterns if not any(fnmatch.fnmatch(n, p) for n in nodes)]
    if unmatched:
        raise DCPFDataError(
            f"{LOG_PREFIX} kvl_nodes patterns matched no node: {unmatched}. "
            f"Available nodes: {sorted(nodes)}"
        )
    return sorted(selected)


def _select_technologies(optimization_setup, impedance_file):
    """Find the transport technologies that carry impedance data.

    :param optimization_setup: the OptimizationSetup
    :param impedance_file: name of the impedance input file
    :return: list of TransportTechnology objects
    """
    transport_techs = optimization_setup.get_all_elements(TransportTechnology)
    requested = config.get("technologies")
    if requested:
        by_name = {t.name: t for t in transport_techs}
        missing = [n for n in requested if n not in by_name]
        if missing:
            raise DCPFDataError(
                f"{LOG_PREFIX} configured technologies not found: {missing}. "
                f"Available: {sorted(by_name)}"
            )
        return [by_name[n] for n in requested]
    # auto-detect: every transport technology that declares an impedance attribute
    return [
        t for t in transport_techs if impedance_file in t.data_input.attribute_dict
    ]


def _read_impedance(tech, impedance_file):
    """Read raw impedances for one technology, without validating them.

    Validation and the inversion to susceptance are deferred to
    :func:`_build_lines`, once the genuine KVL lines are known: an impedance
    that is zero, negative or non-finite on an edge outside the KVL region (or
    on an edge this technology is not on) is never used, must not abort the
    run, and must never be inverted (``1/0``).

    :param tech: TransportTechnology object
    :param impedance_file: name of the impedance input file
    :return: pandas Series of impedance indexed by edge
    """
    impedance = tech.data_input.extract_input_data(
        file_name=impedance_file, index_sets=["set_edges"], unit_category={}
    )
    # extract_input_data returns a single-level MultiIndex; flatten it so that
    # .loc[edge] yields a scalar rather than a one-element Series
    if impedance.index.nlevels == 1 and isinstance(impedance.index, pd.MultiIndex):
        impedance.index = impedance.index.get_level_values(0)
    return impedance


def _supplied_impedance_edges(tech, impedance_file):
    """Edge ids for which the impedance input file actually supplies a value.

    ``extract_input_data`` fills every edge missing from the file with the
    ``attributes.json`` default, so the processed series cannot tell real data
    from the fallback — the exact hazard the module docstring calls out.
    Re-reading the raw CSV (scenario-aware, including a scenario part file)
    recovers that distinction, so :func:`_build_lines` can warn about KVL
    lines that silently run on the default.

    :param tech: TransportTechnology object
    :param impedance_file: name of the impedance input file
    :return: set of edge ids present in the raw file(s), or None when the raw
        file cannot be resolved (the defaulted-line check is then skipped)
    """
    data_input = tech.data_input
    try:
        scenario_dict = getattr(data_input, "scenario_dict", None)
        f_name = impedance_file
        part_name = None
        if scenario_dict is not None:
            f_name, _ = scenario_dict.get_param_file(tech.name, impedance_file)
            part_name = scenario_dict.get_param_part_file(tech.name, impedance_file)
        df_raw = data_input.read_input_csv(f_name)
        if df_raw is None:
            return None
        index_names = getattr(data_input, "index_names", None)
        edge_header = (
            index_names["set_edges"] if index_names is not None else "edge"
        )
        if edge_header not in df_raw.columns:
            return None
        supplied = {str(e) for e in df_raw[edge_header]}
        if part_name is not None:
            df_part = data_input.read_input_csv(part_name)
            if df_part is not None and edge_header in df_part.columns:
                supplied |= {str(e) for e in df_part[edge_header]}
        return supplied
    except Exception as exc:  # diagnostic only — never abort the run over it
        logging.debug(
            f"{LOG_PREFIX} technology '{tech.name}': could not read the raw "
            f"'{impedance_file}' file to detect defaulted impedances ({exc}); "
            f"skipping that check."
        )
        return None


def _edges_absent_in_all_years(optimization_setup, tech_name, edges):
    """Edges where a technology cannot exist in any year, by its capacity limit.

    ZEN-garden creates a flow variable for every (technology, edge) combination
    and expresses absence through capacity rather than through the index sets —
    ``capacity_limit`` of zero is the idiom, as in
    ``tests/testcases/test_2b``. ``capacity_limit`` rather than
    ``capacity_existing`` is read here because a zero limit means the corridor
    can never carry the technology, whereas zero existing capacity merely means
    it has not been built *yet*. An expandable corridor keeps its KVL line, but
    that is a hazard, not a virtue: if the optimizer leaves it unbuilt, the
    hard KVL equality degenerates to ``theta_u == theta_v`` — see
    :func:`_investable_edges` and the ``investment_mode`` config key, which
    report exactly these corridors.

    This is the "absent in *every* year" test: such an edge is dropped entirely,
    because its impedance is a defaulted placeholder rather than real data. It
    is deliberately not the whole story — an edge whose limit is zero in some
    years and positive in others is a real KVL line whose constraint must simply
    be suspended in the zero-limit years, which :func:`_absent_years` and the
    constraint mask in :func:`after_optimization_construction` take care of.

    :param optimization_setup: the OptimizationSetup
    :param tech_name: name of the transport technology
    :param edges: edges to test
    :return: set of edges where the technology has no capacity in any year
    """
    capacity_limit = getattr(optimization_setup.parameters, "capacity_limit", None)
    if capacity_limit is None:
        return set()
    limit = capacity_limit.sel(
        {"set_technologies": tech_name, "set_capacity_types": "power"}
    )
    limit = limit.sel({"set_location": [e for e in edges if e in limit.coords["set_location"]]})
    # Absent only when no year allows any capacity at all
    largest = limit.max("set_time_steps_yearly")
    return {
        str(edge)
        for edge, value in zip(
            np.asarray(largest.coords["set_location"].data),
            np.asarray(largest.data, dtype=float),
            strict=True,
        )
        if not value > 0
    }


def _absent_years(optimization_setup, tech_name, edges):
    """Per-year absence mask for a technology's edges, by its capacity limit.

    Complements :func:`_edges_absent_in_all_years`: an edge whose
    ``capacity_limit`` is zero only in *some* years (a corridor that does not
    exist yet) keeps its KVL line, but the constraint must not be imposed in
    the zero-limit years — there the core constraints pin ``capacity`` and
    ``flow`` to zero, and the KVL equality would degenerate to
    ``theta_u == theta_v``, a zero-impedance tie between buses that are not
    yet connected.

    :param optimization_setup: the OptimizationSetup
    :param tech_name: name of the transport technology
    :param edges: edges to test
    :return: boolean DataArray over (set_edges, set_time_steps_yearly), True
        where the technology cannot exist; None when no capacity limit is known
    """
    capacity_limit = getattr(optimization_setup.parameters, "capacity_limit", None)
    if capacity_limit is None:
        return None
    limit = capacity_limit.sel(
        {"set_technologies": tech_name, "set_capacity_types": "power"}
    )
    available = [e for e in edges if e in limit.coords["set_location"]]
    # NaN compares False and therefore counts as absent, matching
    # _edges_absent_in_all_years
    absent = ~(limit.sel({"set_location": available}) > 0)
    # Drop the scalar technology/capacity-type coords so they cannot clash when
    # the mask is aligned against the constraint expression later on.
    absent = absent.rename({"set_location": "set_edges"}).reset_coords(drop=True)
    if len(available) < len(edges):
        # An edge the parameter does not know cannot be ruled out — treat as present
        absent = absent.reindex({"set_edges": list(edges)}, fill_value=False)
    return absent


def _investable_edges(optimization_setup, tech_name, edges):
    """Edges where the optimizer decides whether the technology gets built.

    An edge is investable when there is a year in which the corridor may exist
    (``capacity_limit > 0``) but no capacity exists yet
    (``existing_capacities`` not positive; NaN counts as zero, matching
    :func:`_edges_absent_in_all_years`) — unless capacity additions are
    forbidden outright (``capacity_addition_max == 0``), in which case the
    capacity is fixed and there is no investment decision. These are the same
    parameters the core lifetime/limit constraints are built from.

    On such an edge the KVL equality is a modelling hazard: if the optimizer
    leaves it at zero capacity, the capacity-factor constraint pins the flow to
    zero and KVL degenerates to ``theta_u == theta_v`` — the unbuilt line
    becomes a zero-impedance short instead of an open circuit. Handling that
    correctly needs a disjunctive formulation; this plugin only detects and
    reports it, per the ``investment_mode`` config key.

    :param optimization_setup: the OptimizationSetup
    :param tech_name: name of the transport technology
    :param edges: edges to test
    :return: set of investable edges
    """
    parameters = optimization_setup.parameters
    capacity_limit = getattr(parameters, "capacity_limit", None)
    existing_capacities = getattr(parameters, "existing_capacities", None)
    if capacity_limit is None or existing_capacities is None:
        return set()
    capacity_addition_max = getattr(parameters, "capacity_addition_max", None)
    if capacity_addition_max is not None:
        try:
            addition_max = float(
                capacity_addition_max.sel(
                    {"set_technologies": tech_name, "set_capacity_types": "power"}
                )
            )
        except (KeyError, TypeError, ValueError):
            addition_max = None
        if addition_max == 0:
            # Additions are forbidden, so capacity stays at its existing value
            # in every year — nothing to invest in.
            return set()
    limit = capacity_limit.sel(
        {"set_technologies": tech_name, "set_capacity_types": "power"}
    )
    existing = existing_capacities.sel(
        {"set_technologies": tech_name, "set_capacity_types": "power"}
    )
    available = [
        e
        for e in edges
        if e in limit.coords["set_location"] and e in existing.coords["set_location"]
    ]
    if not available:
        return set()
    limit = limit.sel({"set_location": available})
    existing = existing.sel({"set_location": available})
    # Investable in a year when the corridor may exist but nothing exists yet;
    # NaN existing capacity compares False and therefore counts as zero.
    investable = ((limit > 0) & ~(existing > 0)).any("set_time_steps_yearly")
    return {
        str(edge)
        for edge, value in zip(
            np.asarray(investable.coords["set_location"].data),
            np.asarray(investable.data, dtype=bool),
            strict=True,
        )
        if value
    }


def _check_investable_corridors(optimization_setup, lines_by_tech):
    """Detect KVL corridors whose capacity is an open investment decision.

    KVL is imposed as a hard equality, so a corridor the optimizer leaves
    unbuilt degenerates to a zero-impedance tie (see :func:`_investable_edges`).
    ``investment_mode`` decides what to do about it: ``"warn"`` (default) logs
    the corridors and proceeds, ``"error"`` raises so the hazard cannot pass
    silently. Called before any constraint is added, so ``"error"`` leaves the
    model untouched.

    :param optimization_setup: the OptimizationSetup
    :param lines_by_tech: dict tech name -> line description
    """
    mode = config.get("investment_mode", "warn")
    if mode not in ("warn", "error"):
        raise DCPFDataError(
            f"{LOG_PREFIX} investment_mode must be 'warn' or 'error', got "
            f"'{mode}'. A disjunctive (big-M) treatment of unbuilt corridors is "
            f"not implemented."
        )
    for tech_name, lines in lines_by_tech.items():
        investable = _investable_edges(
            optimization_setup, tech_name, lines["fwd"] + lines["rev"]
        )
        if not investable:
            continue
        corridors = sorted(
            line_id
            for line_id, fwd, rev in zip(
                lines["line_ids"], lines["fwd"], lines["rev"], strict=True
            )
            if fwd in investable or rev in investable
        )
        if not corridors:
            continue
        message = (
            f"technology '{tech_name}': {len(corridors)} KVL corridor(s) have "
            f"no existing capacity but allow investment: {corridors[:10]}"
            f"{' ...' if len(corridors) > 10 else ''}. KVL is a hard equality, "
            f"so a corridor left unbuilt degenerates to theta_u == theta_v — a "
            f"zero-impedance short that can force investment the network does "
            f"not need. Fix the corridor capacities via capacity_existing, "
            f"exclude the corridor from the KVL region, or accept the risk."
        )
        if mode == "error":
            raise DCPFDataError(
                f"{LOG_PREFIX} {message} (investment_mode='error'; set it to "
                f"'warn' to proceed anyway.)"
            )
        logging.warning(f"{LOG_PREFIX} {message}")


def _build_lines(
    tech,
    impedance,
    kvl_nodes,
    nodes_on_edges,
    valid_edges,
    reverse_by_edge,
    absent_edges,
    supplied_edges,
):
    """Pair the directed edges of one technology into undirected KVL lines.

    An edge qualifies when both of its endpoints are in the KVL node set. Its
    reverse partner must exist, because the constraint is written on the net
    flow of the pair. Parallel corridors between the same two nodes are kept as
    separate lines, each paired with its own reverse edge and carrying its own
    susceptance — parallel susceptances then add up through the shared voltage
    angles, which is what makes an aggregated corridor behave correctly without
    a blended equivalent impedance.

    Impedance is validated here, on the genuine KVL lines only, so a zero or
    defaulted impedance on an edge outside the KVL region (or one this
    technology is not on) cannot abort the run. Both directions of each line
    are read and must agree — a direction missing from the impedance file
    silently receives the ``attributes.json`` default, and reading only one
    side would let that fallback shadow the real data. A line whose directions
    disagree is a data error; a line supplied in *neither* direction (detected
    from the raw file via ``supplied_edges``) runs on the default and is
    warned about, since the file cannot mark absence.

    :param tech: TransportTechnology object
    :param impedance: raw impedance per edge (validated here, then inverted)
    :param kvl_nodes: set of nodes inside the KVL region
    :param nodes_on_edges: dict edge -> (node_from, node_to)
    :param valid_edges: edges for which this technology has a flow variable
    :param reverse_by_edge: dict edge -> reverse edge (or None)
    :param absent_edges: edges where this technology has no capacity in any year
    :param supplied_edges: edge ids actually present in the raw impedance
        file, or None to skip the defaulted-line warning
    :return: dict of parallel lists describing the lines
    """
    kvl_nodes = set(kvl_nodes)

    line_ids, fwd, rev, from_nodes, to_nodes, b_values = [], [], [], [], [], []
    missing_reverse, missing_variable, self_loops, absent = [], [], [], []
    bad_impedance, mismatched, defaulted = [], [], []

    for edge, (u, v) in nodes_on_edges.items():
        if u not in kvl_nodes or v not in kvl_nodes:
            continue  # boundary or fully external edge -> stays a transport edge
        if u == v:
            self_loops.append(edge)
            continue
        reverse = reverse_by_edge.get(edge)
        if not absent_edges.isdisjoint((edge, reverse)):
            # ZEN-garden builds a flow variable for every (technology, edge)
            # pair and expresses "this technology is not here" through a zero
            # capacity limit, so an edge belonging to another technology still
            # reaches this loop -- and reads a defaulted impedance rather than a
            # real one. Skipping is not merely tidier: a KVL constraint on an
            # edge whose flow is pinned to zero would read 0 = B * dtheta and
            # force the two buses to an identical angle, distorting every
            # parallel path in the network. Checked before the reverse-partner
            # test so a corridor the technology is not on cannot abort the run.
            # Only edges absent in *every* year are dropped here; an edge with
            # a zero limit in some years keeps its line, and the per-year mask
            # built from _absent_years suspends its constraint in those years.
            absent.append(edge)
            continue
        if reverse is None:
            # Checked before the canonical-orientation skip: a one-way corridor
            # stored as '<v>-<u>' with v > u has no canonical partner to handle
            # it, and must fail loudly rather than silently lose its KVL line.
            missing_reverse.append(edge)
            continue
        if u > v:
            continue  # handled from its canonical partner
        if edge not in valid_edges or reverse not in valid_edges:
            missing_variable.append(edge)
            continue
        # Validate only here, on a genuine KVL line, and read both directions:
        # a direction missing from the impedance file carries the
        # attributes.json default, and reading only the canonical side would
        # let that fallback silently shadow the supplied value.
        z_fwd = float(impedance.loc[edge])
        z_rev = float(impedance.loc[reverse])
        bad = [e for e, z in ((edge, z_fwd), (reverse, z_rev))
               if not (np.isfinite(z) and z > 0)]
        if bad:
            bad_impedance.extend(bad)
            continue
        if not np.isclose(z_fwd, z_rev):
            mismatched.append((edge, z_fwd, reverse, z_rev))
            continue
        if (
            supplied_edges is not None
            and edge not in supplied_edges
            and reverse not in supplied_edges
        ):
            defaulted.append(edge)
        line_ids.append(edge)
        fwd.append(edge)
        rev.append(reverse)
        from_nodes.append(u)
        to_nodes.append(v)
        b_values.append(1.0 / z_fwd)

    if self_loops:
        logging.warning(
            f"{LOG_PREFIX} technology '{tech.name}': ignoring {len(self_loops)} "
            f"self-loop edge(s) inside the KVL region: {sorted(self_loops)[:5]}"
        )
    if absent:
        logging.info(
            f"{LOG_PREFIX} technology '{tech.name}': {len(absent)} edge(s) inside "
            f"the KVL region carry no capacity for it and are left to whichever "
            f"technology owns them: {sorted(absent)[:5]}"
            f"{' ...' if len(absent) > 5 else ''}"
        )
    if missing_reverse:
        raise DCPFDataError(
            f"{LOG_PREFIX} technology '{tech.name}': edges {sorted(missing_reverse)} "
            f"lie inside the KVL region but have no unambiguous reverse edge. KVL is "
            f"imposed on the net flow of a directed pair, so each edge needs exactly "
            f"one partner. Add the missing reverse edge to set_edges.csv, or give "
            f"parallel corridors distinct '<from>-<to>__<asset>' ids so they can be "
            f"told apart."
        )
    if missing_variable:
        raise DCPFDataError(
            f"{LOG_PREFIX} technology '{tech.name}': edges {sorted(missing_variable)} "
            f"lie inside the KVL region but carry no flow variable for this "
            f"technology. Either exclude their nodes from kvl_nodes or make the "
            f"technology available on those edges."
        )
    if bad_impedance:
        raise DCPFDataError(
            f"{LOG_PREFIX} technology '{tech.name}' has non-positive or "
            f"non-finite impedance on KVL edges {sorted(bad_impedance)}. "
            f"Every edge inside the KVL region needs a positive, finite impedance."
        )
    if mismatched:
        detail = ", ".join(
            f"{e}={z_f:g} vs {r}={z_r:g}"
            for e, z_f, r, z_r in sorted(mismatched)
        )
        raise DCPFDataError(
            f"{LOG_PREFIX} technology '{tech.name}': the two directions of "
            f"{len(mismatched)} KVL corridor(s) disagree on impedance: {detail}. "
            f"KVL treats the directed pair as one physical line with one "
            f"impedance. A direction missing from the impedance file silently "
            f"receives the attributes.json default, which is the usual cause — "
            f"supply the same value for both directions."
        )
    if defaulted:
        default_value = float(impedance.loc[defaulted[0]])
        logging.warning(
            f"{LOG_PREFIX} technology '{tech.name}': {len(defaulted)} KVL "
            f"line(s) appear in no impedance input file and run on the "
            f"attributes.json default impedance ({default_value:g}): "
            f"{sorted(defaulted)[:5]}{' ...' if len(defaulted) > 5 else ''}"
        )
    return {
        "line_ids": line_ids,
        "fwd": fwd,
        "rev": rev,
        "from_nodes": from_nodes,
        "to_nodes": to_nodes,
        "b": b_values,
    }


def _signed_flow_admissible(optimization_setup, tech_name, edges):
    """Check whether a technology's flow on the KVL edges may go negative.

    Losses, variable opex and emissions are tied by equality to a non-negative
    variable times the flow, and a non-zero ``min_load`` places the technology
    in ``set_on_off``, whose constraints force
    ``flow_transport >= min_load * capacity_on_off_helper_var >= 0``. Each of
    these forbids a negative flow, so the signed representation is admissible
    only when all of them are absent on the KVL edges.

    :param optimization_setup: the OptimizationSetup
    :param tech_name: name of the transport technology
    :param edges: edges to check
    :return: (bool, list of offending parameter names)
    """
    parameters = optimization_setup.parameters
    offenders = []
    for name in (
        "transport_loss_factor",
        "opex_specific_variable",
        "carbon_intensity_technology",
    ):
        param = getattr(parameters, name, None)
        if param is None:
            continue
        # Reduce over any time dimension before selecting edges. Selecting first
        # would materialise an (edges x time steps) copy of the parameter — over
        # 100 MB at European resolution — merely to answer a yes/no question,
        # whereas reducing a single-label view leaves one value per location.
        by_location = param.loc[tech_name]
        if "set_time_steps_operation" in by_location.dims:
            extremes = (
                by_location.max("set_time_steps_operation"),
                by_location.min("set_time_steps_operation"),
            )
        else:
            extremes = (by_location,)
        # NaN propagates through max/min and compares False, so it counts as zero
        if any(
            float(extreme.loc[edges].max()) > 0 or float(extreme.loc[edges].min()) < 0
            for extreme in extremes
        ):
            offenders.append(name)
    # A non-zero min_load places the technology in set_on_off
    # (Element.check_on_off_modeled) and the on-off constraints bound the flow
    # below by min_load * capacity_on_off_helper_var >= 0 — contradicting a
    # signed flow. The hook runs after construction, so the realised model is
    # exact ground truth: tech_on_var is removed again when no on-off rows were
    # emitted, hence its presence with a valid (non--1) label on these edges
    # means the on-off constraints exist there.
    model = getattr(optimization_setup, "model", None)
    model_variables = getattr(model, "variables", None)
    if model_variables is not None and "tech_on_var" in model_variables:
        labels = model_variables["tech_on_var"].labels
        if tech_name in labels.coords["set_technologies"]:
            by_tech = labels.sel({"set_technologies": tech_name})
            on_edges = [e for e in edges if e in by_tech.coords["set_location"]]
            if on_edges and bool(
                (by_tech.sel({"set_location": on_edges}) != -1).any()
            ):
                offenders.append(
                    "min_load (on-off constraints exist on these edges)"
                )
    return (not offenders), offenders


def _resolve_flow_representation(optimization_setup, tech_name, edges):
    """Decide whether a technology uses the signed or the net formulation.

    :param optimization_setup: the OptimizationSetup
    :param tech_name: name of the transport technology
    :param edges: the directed edges that will be KVL-constrained
    :return: "signed" or "net"
    """
    requested = config.get("flow_representation", "auto")
    if requested not in ("auto", "signed", "net"):
        raise DCPFDataError(
            f"{LOG_PREFIX} flow_representation must be 'auto', 'signed' or 'net', "
            f"got '{requested}'."
        )
    if requested == "net":
        return "net"

    ok, offenders = _signed_flow_admissible(optimization_setup, tech_name, edges)
    if requested == "signed" and not ok:
        raise DCPFDataError(
            f"{LOG_PREFIX} flow_representation='signed' requires a lossless, "
            f"cost-free line without on-off behaviour, but technology "
            f"'{tech_name}' has non-zero {offenders} on KVL edges. Losses, "
            f"variable opex and emissions are tied by equality to a "
            f"non-negative variable times the flow, and on-off (min_load) "
            f"constraints bound the flow below by zero, so a signed flow would "
            f"make the model infeasible. Set them to zero or use "
            f"flow_representation='net'."
        )
    if ok:
        return "signed"
    logging.info(
        f"{LOG_PREFIX} technology '{tech_name}' has non-zero {offenders}; using "
        f"the net-flow formulation. Directed flows may circulate — call "
        f"report_circulation() after solving."
    )
    return "net"


def _warn_on_bypass_paths(
    optimization_setup,
    lines_by_tech,
    kvl_nodes,
    nodes_on_edges,
    node_component,
    absent_by_tech,
):
    """Warn about controllable corridors parallel to the KVL subgraph.

    A transport flow whose endpoints sit in the same synchronous component but
    which is not itself KVL-constrained lets flow route around the impedance
    constraint. That is legitimate for a genuine HVDC embedded in an AC grid,
    and a modelling error otherwise, so it is reported rather than rejected.

    KVL membership is a *(technology, edge)* property: every transport
    technology owns a flow variable on every edge, so a second technology on a
    KVL-constrained corridor is exactly the parallel path this check exists to
    flag — testing the edge id alone would miss it. Candidates are restricted
    to transport technologies sharing a reference carrier with the KVL
    technologies (a hydrogen pipeline over an electrical corridor is not an
    electrical parallel path), and pairs where the technology has no capacity
    on the edge in any year are skipped, since their flow is pinned to zero.
    The two directions of a corridor are deduplicated with an
    orientation-independent key, so a one-way corridor stored against either
    lexicographic orientation is still seen.

    :param optimization_setup: the OptimizationSetup
    :param lines_by_tech: dict tech name -> line description
    :param kvl_nodes: set of nodes inside the KVL region
    :param nodes_on_edges: dict edge -> (node_from, node_to)
    :param node_component: dict node -> component id
    :param absent_by_tech: dict tech name -> edges absent in every year;
        extended in place for candidate technologies not yet in it
    """
    constrained = {
        (tech_name, edge)
        for tech_name, d in lines_by_tech.items()
        for edge in d["fwd"] + d["rev"]
    }
    transport_techs = optimization_setup.get_all_elements(TransportTechnology)
    by_name = {t.name: t for t in transport_techs}
    kvl_carriers = {
        carrier
        for tech_name in lines_by_tech
        if tech_name in by_name
        for carrier in by_name[tech_name].reference_carrier
    }
    candidates = [
        t
        for t in transport_techs
        if any(carrier in kvl_carriers for carrier in t.reference_carrier)
    ]

    bypass = []
    corridors = set()
    for tech in candidates:
        absent_edges = absent_by_tech.get(tech.name)
        if absent_edges is None:
            absent_edges = _edges_absent_in_all_years(
                optimization_setup, tech.name, list(nodes_on_edges)
            )
            absent_by_tech[tech.name] = absent_edges
        seen = set()
        for edge, (u, v) in nodes_on_edges.items():
            if u == v or u not in kvl_nodes or v not in kvl_nodes:
                continue
            if edge in absent_edges:
                # The flow is pinned to zero here, so nothing can route over
                # it. Checked before the corridor dedup so the other, present
                # direction of a half-absent corridor is still examined.
                continue
            corridor = pair_key(edge, min(u, v), max(u, v))
            if corridor in seen:
                continue
            seen.add(corridor)
            if (tech.name, edge) in constrained:
                continue
            if (
                node_component.get(u) is not None
                and node_component.get(u) == node_component.get(v)
            ):
                bypass.append(f"{tech.name}:{edge}")
                corridors.add(corridor)
    if bypass:
        logging.warning(
            f"{LOG_PREFIX} {len(corridors)} corridor(s) inside a synchronous "
            f"component carry a transport flow that is not KVL-constrained and "
            f"can route around the impedance constraint "
            f"({len(bypass)} technology-corridor pair(s)): {sorted(bypass)[:10]}"
            f"{' ...' if len(bypass) > 10 else ''}. This is correct for embedded "
            f"HVDC and a data error otherwise."
        )


def _components(line_nodes_from, line_nodes_to, nodes):
    """Find the connected components of the KVL subgraph.

    :param line_nodes_from: from-node of every KVL line
    :param line_nodes_to: to-node of every KVL line
    :param nodes: all nodes of the system
    :return: (dict node -> component id, number of components)
    """
    index = {n: i for i, n in enumerate(nodes)}
    rows = [index[u] for u in line_nodes_from]
    cols = [index[v] for v in line_nodes_to]
    data = np.ones(len(rows))
    adjacency = coo_matrix(
        (data, (rows, cols)), shape=(len(nodes), len(nodes))
    ).tocsr()
    n_comp, labels = connected_components(adjacency, directed=False)
    touched = set(line_nodes_from) | set(line_nodes_to)
    node_component = {n: int(labels[index[n]]) for n in touched}
    used = sorted(set(node_component.values()))
    return node_component, used


def _resolve_slack_nodes(node_component, used_components):
    """Pick one reference bus per synchronous component.

    :param node_component: dict node -> component id
    :param used_components: component ids that contain at least one line
    :return: list of slack node names
    """
    configured = config.get("slack_nodes") or []
    by_component = {}
    for node in sorted(node_component):
        by_component.setdefault(node_component[node], []).append(node)

    if configured:
        chosen = {}
        for node in configured:
            if node not in node_component:
                raise DCPFDataError(
                    f"{LOG_PREFIX} configured slack node '{node}' is not part of any "
                    f"KVL component. Slack nodes must lie inside the KVL region."
                )
            comp = node_component[node]
            if comp in chosen:
                raise DCPFDataError(
                    f"{LOG_PREFIX} two slack nodes configured for the same "
                    f"component: '{chosen[comp]}' and '{node}'."
                )
            chosen[comp] = node
        missing = [c for c in used_components if c not in chosen]
        if missing:
            raise DCPFDataError(
                f"{LOG_PREFIX} no slack node configured for component(s) {missing}. "
                f"Each synchronous component needs exactly one. Candidates: "
                f"{ {c: by_component[c][0] for c in missing} }"
            )
        return [chosen[c] for c in sorted(chosen)]

    # default: the alphabetically first node of each component, for reproducibility
    return [by_component[c][0] for c in used_components]


def _set_signed_bounds(flow, tech_name, lines):
    """Free the forward edge to go negative and remove the reverse edge.

    The forward edge carries the signed flow, so its negative half is transfer
    in the reverse direction and must be rated by the *reverse* edge's upper
    bound: ``lower[fwd] = -upper[rev]`` (read before the reverse bounds are
    zeroed). Zero-width bounds are strictly better than an equality constraint:
    the solver eliminates the reverse variables in presolve, so they add no rows.

    :param flow: the flow_transport variable
    :param tech_name: name of the transport technology
    :param lines: line description produced by :func:`_build_lines`
    """
    fwd_sel = {"set_transport_technologies": tech_name, "set_edges": lines["fwd"]}
    rev_sel = {"set_transport_technologies": tech_name, "set_edges": lines["rev"]}
    valid_fwd = flow.labels.loc[fwd_sel] != -1
    # relabel the reverse-edge selections onto the forward edge ids so xarray
    # aligns them position-by-position with the forward selection
    valid_rev = (flow.labels.loc[rev_sel] != -1).assign_coords(
        {"set_edges": lines["fwd"]}
    )
    upper_rev = flow.upper.loc[rev_sel].assign_coords({"set_edges": lines["fwd"]})
    flow.lower.loc[fwd_sel] = xr.where(
        valid_fwd & valid_rev, -upper_rev, flow.lower.loc[fwd_sel]
    )
    flow.lower.loc[rev_sel] = xr.where(valid_rev.data, 0.0, flow.lower.loc[rev_sel])
    flow.upper.loc[rev_sel] = xr.where(valid_rev.data, 0.0, flow.upper.loc[rev_sel])


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


@Events.register(Event.after_optimization_construction)
def after_optimization_construction(optimization_setup, **kwargs):
    """Add DC power flow constraints to the constructed model.

    :param optimization_setup: the OptimizationSetup the plugin operates on
    """
    if not enabled:
        return
    model = optimization_setup.model
    verbose = bool(config.get("verbose", False))
    impedance_file = config.get("impedance_file", "impedance")

    techs = _select_technologies(optimization_setup, impedance_file)
    if not techs:
        logging.warning(
            f"{LOG_PREFIX} no transport technology declares an '{impedance_file}' "
            f"attribute — skipping."
        )
        return

    if "flow_transport" not in model.variables:
        logging.warning(f"{LOG_PREFIX} no flow_transport variable — skipping.")
        return

    nodes = list(optimization_setup.sets["set_nodes"])
    nodes_on_edges = optimization_setup.energy_system.set_nodes_on_edges
    flow = model.variables["flow_transport"]
    valid_edges = set(np.asarray(flow.coords["set_edges"].data).tolist())

    kvl_nodes = _resolve_kvl_nodes(config.get("kvl_nodes"), nodes)

    # ---- pair directed edges once, and refuse to guess ---------------------
    reverse_by_edge, ambiguous_edges = build_reverse_edge_map(nodes_on_edges)
    ambiguous_in_region = {
        e
        for e in ambiguous_edges
        if nodes_on_edges[e][0] in set(kvl_nodes)
        and nodes_on_edges[e][1] in set(kvl_nodes)
    }
    if ambiguous_in_region:
        raise DCPFDataError(
            f"{LOG_PREFIX} cannot pair directed edges inside the KVL region. "
            f"{describe_ambiguous_edges(ambiguous_in_region, nodes_on_edges)}. "
            f"Pairing on the node pair alone would attach every parallel corridor "
            f"to the same reverse edge and leave the rest unconstrained."
        )
    if ambiguous_edges - ambiguous_in_region:
        logging.info(
            f"{LOG_PREFIX} {len(ambiguous_edges - ambiguous_in_region)} ambiguous "
            f"edge pair(s) outside the KVL region — not constrained, so harmless here."
        )

    # ---- select and pair the lines, per technology -------------------------
    lines_by_tech = {}
    # Computed once per technology and shared with the bypass check, which
    # also needs it for technologies outside the KVL selection.
    absent_by_tech = {}
    for tech in techs:
        impedance = _read_impedance(tech, impedance_file)
        supplied_edges = _supplied_impedance_edges(tech, impedance_file)
        absent_by_tech[tech.name] = _edges_absent_in_all_years(
            optimization_setup, tech.name, list(nodes_on_edges)
        )
        lines = _build_lines(
            tech,
            impedance,
            kvl_nodes,
            nodes_on_edges,
            valid_edges,
            reverse_by_edge,
            absent_by_tech[tech.name],
            supplied_edges,
        )
        if lines["line_ids"]:
            lines_by_tech[tech.name] = lines

    if not lines_by_tech:
        logging.warning(
            f"{LOG_PREFIX} the KVL node set {kvl_nodes} contains no internal line "
            f"— no constraints added."
        )
        return

    # KVL corridors whose capacity is an open investment decision: report (or
    # abort, per investment_mode) before touching the model.
    _check_investable_corridors(optimization_setup, lines_by_tech)

    # ---- synchronous components, over the union of all KVL lines -----------
    all_from = [u for d in lines_by_tech.values() for u in d["from_nodes"]]
    all_to = [v for d in lines_by_tech.values() for v in d["to_nodes"]]
    node_component, used_components = _components(all_from, all_to, nodes)
    slack_nodes = _resolve_slack_nodes(node_component, used_components)
    _warn_on_bypass_paths(
        optimization_setup,
        lines_by_tech,
        set(kvl_nodes),
        nodes_on_edges,
        node_component,
        absent_by_tech,
    )

    # ---- voltage angles, only where they mean something --------------------
    theta_nodes = sorted(node_component)
    times = flow.coords["set_time_steps_operation"]
    theta = model.add_variables(
        lower=-np.inf,
        upper=np.inf,
        coords=[theta_nodes, times.data],
        dims=["set_nodes", "set_time_steps_operation"],
        name="theta",
    )

    # ---- KVL, one vectorized constraint block per technology ---------------
    # Indexers carry the canonical edge as the coordinate and the quantity to be
    # selected as the value, so flows and both angles align on a common
    # set_edges axis without any Python-level loop over edges or time steps.
    time_step_year = xr.DataArray(
        [
            optimization_setup.energy_system.time_steps.convert_time_step_operation2year(
                t
            )
            for t in times.data
        ],
        coords=[times],
    )
    representations = {}

    for tech_name, lines in lines_by_tech.items():
        line_ids = lines["line_ids"]
        coords = {"set_edges": line_ids}

        fwd_idx = xr.DataArray(lines["fwd"], dims="set_edges", coords=coords)
        rev_idx = xr.DataArray(lines["rev"], dims="set_edges", coords=coords)
        from_idx = xr.DataArray(lines["from_nodes"], dims="set_edges", coords=coords)
        to_idx = xr.DataArray(lines["to_nodes"], dims="set_edges", coords=coords)
        susceptance = xr.DataArray(lines["b"], dims="set_edges", coords=coords)

        representation = _resolve_flow_representation(
            optimization_setup, tech_name, lines["fwd"] + lines["rev"]
        )
        representations[tech_name] = representation

        flow_tech = flow.sel({"set_transport_technologies": tech_name})
        angle_difference = theta.sel({"set_nodes": from_idx}) - theta.sel(
            {"set_nodes": to_idx}
        )

        # A corridor whose capacity_limit is zero in some years does not exist
        # in those years: the core constraints pin capacity and flow to zero,
        # and imposing KVL there would degenerate to theta_u == theta_v — a
        # zero-impedance tie between unconnected buses. Suspend the KVL rows in
        # those years via the constraint mask (True keeps a row). A line is
        # absent in a year when either of its directed edges is.
        kvl_mask = None
        absent_years = _absent_years(
            optimization_setup, tech_name, lines["fwd"] + lines["rev"]
        )
        if absent_years is not None:
            # Select by list and relabel explicitly: with a DataArray indexer
            # named like the indexed dim, xarray keeps the *selected* labels as
            # the result's coordinate, so the reverse selection would come back
            # on reverse-edge ids and misalign against the line axis.
            absent_line_years = absent_years.sel(
                {"set_edges": lines["fwd"]}
            ).assign_coords({"set_edges": line_ids}) | absent_years.sel(
                {"set_edges": lines["rev"]}
            ).assign_coords({"set_edges": line_ids})
            if bool(absent_line_years.any()):
                kvl_mask = ~absent_line_years.sel(
                    {"set_time_steps_yearly": time_step_year}
                ).drop_vars("set_time_steps_yearly", errors="ignore")
                n_masked_lines = int((absent_line_years.any("set_time_steps_yearly")).sum())
                logging.info(
                    f"{LOG_PREFIX} technology '{tech_name}': {n_masked_lines} "
                    f"line(s) have no capacity in some year(s); their KVL rows "
                    f"are dropped in those years."
                )

        if representation == "signed":
            # The reverse edge is removed from the problem and the forward edge
            # carries the signed flow, so the solution is unique.
            _set_signed_bounds(flow, tech_name, lines)
            lhs = flow_tech.sel({"set_edges": fwd_idx}) - susceptance * angle_difference
            model.add_constraints(lhs == 0, name=f"dclf_kvl_{tech_name}", mask=kvl_mask)
            # ZEN-garden's stock capacity constraint is one-sided
            # (flow <= max_load * capacity); a signed flow needs the mirror image.
            # The negative half is transfer in the reverse direction, so it is
            # rated by the *reverse* edge's max_load and capacity. Selecting with
            # rev_idx (dims "set_edges", coords line_ids) relabels the result
            # onto the line axis, aligning it with flow_tech on fwd_idx.
            term_capacity = (
                optimization_setup.parameters.max_load.sel(
                    {"set_technologies": tech_name, "set_location": rev_idx}
                )
                * model.variables["capacity"].sel(
                    {
                        "set_technologies": tech_name,
                        "set_capacity_types": "power",
                        "set_location": rev_idx,
                        "set_time_steps_yearly": time_step_year,
                    }
                )
            )
            model.add_constraints(
                flow_tech.sel({"set_edges": fwd_idx}) + term_capacity >= 0,
                name=f"dclf_capacity_reverse_{tech_name}",
            )
        else:
            net_flow = flow_tech.sel({"set_edges": fwd_idx}) - flow_tech.sel(
                {"set_edges": rev_idx}
            )
            lhs = net_flow - susceptance * angle_difference
            model.add_constraints(lhs == 0, name=f"dclf_kvl_{tech_name}", mask=kvl_mask)

    # ---- one reference bus per synchronous component -----------------------
    model.add_constraints(
        theta.sel({"set_nodes": slack_nodes}) == 0, name="dclf_reference_bus"
    )

    n_lines = sum(len(d["line_ids"]) for d in lines_by_tech.values())
    n_boundary = sum(
        1
        for edge, (u, v) in nodes_on_edges.items()
        if u < v and (u in set(kvl_nodes)) != (v in set(kvl_nodes))
    )
    logging.info(
        f"{LOG_PREFIX} KVL on {n_lines} line(s) across "
        f"{len(lines_by_tech)} technology(ies), {len(theta_nodes)} bus(es), "
        f"{len(used_components)} synchronous component(s); slack at {slack_nodes}; "
        f"{n_boundary} boundary corridor(s) left as transport; "
        f"flow representation {representations}."
    )
    if verbose:
        for tech_name, lines in lines_by_tech.items():
            logging.info(
                f"{LOG_PREFIX} [{tech_name}] lines: "
                f"{list(zip(lines['from_nodes'], lines['to_nodes'], strict=False))}"
            )
        logging.info(f"{LOG_PREFIX} components: {node_component}")


# --------------------------------------------------------------------------- #
# Post-solve diagnostic
# --------------------------------------------------------------------------- #


def report_circulation(optimization_setup, tolerance=1e-6):
    """Report directed pairs that carry flow in both directions simultaneously.

    Physically only the net flow matters, so a circulating pair is harmless, but
    it indicates that neither losses nor variable opex are penalising it. Non-zero
    results mean reported per-direction flows should be netted before use.

    Call after solving::

        from zen_garden.plugins.dclf.plugin import report_circulation
        report_circulation(optimization_setup)

    :param optimization_setup: the solved OptimizationSetup
    :param tolerance: flow below this magnitude counts as zero
    :return: dict with the maximum and the total circulating flow
    """
    flow = optimization_setup.model.variables["flow_transport"].solution
    nodes_on_edges = optimization_setup.energy_system.set_nodes_on_edges
    reverse_by_edge, ambiguous_edges = build_reverse_edge_map(nodes_on_edges)
    if ambiguous_edges:
        logging.warning(
            f"{LOG_PREFIX} skipping ambiguous edges in the circulation check. "
            f"{describe_ambiguous_edges(ambiguous_edges, nodes_on_edges)}"
        )

    worst, total = 0.0, 0.0
    for edge, (u, v) in nodes_on_edges.items():
        reverse = reverse_by_edge.get(edge)
        if u >= v or reverse is None:
            continue
        if edge not in flow.coords["set_edges"] or reverse not in flow.coords[
            "set_edges"
        ]:
            continue
        pair_min = np.minimum(
            flow.sel(set_edges=edge).data, flow.sel(set_edges=reverse).data
        )
        pair_min = np.where(pair_min > tolerance, pair_min, 0.0)
        worst = max(worst, float(np.max(pair_min, initial=0.0)))
        total += float(np.sum(pair_min))

    if worst > tolerance:
        logging.warning(
            f"{LOG_PREFIX} circulating flow detected — max {worst:.4g}, "
            f"total {total:.4g}. Net the directed flows before reporting."
        )
    else:
        logging.info(f"{LOG_PREFIX} no circulating flow above {tolerance:g}.")
    return {"max": worst, "total": total}


def report_angle_spread(optimization_setup, threshold=0.5):
    """Report the largest voltage angle difference across any KVL line.

    DC power flow replaces ``sin(dtheta)`` with ``dtheta``, which is accurate
    while angle differences stay small and degrades as they grow — the error is
    about 2% at 0.3 rad and 8% at 0.7 rad. Nothing in the model enforces that,
    because the capacity limits bound each ``dtheta = f / B`` only indirectly, so
    a heavily loaded long corridor can quietly leave the range where the
    linearisation is a good approximation.

    This is a diagnostic, not a constraint. Imposing an angle limit would be a
    security assumption on top of DC OPF, and it would tighten the system in a
    way that changes prices; measuring the spread does neither.

    The angles are only interpretable as radians when the exported impedances are
    per-unit on a base matching the model's power unit. With impedances in ohms
    the values are still proportional, so the ranking of the worst corridors
    holds, but the threshold does not mean what it says.

    The lines are recovered from the model's ``dclf_kvl_*`` constraints, because
    angles are only comparable between buses that share a synchronous reference:
    a corridor joining two components would report the difference of two
    unrelated arbitrary references. A model reloaded without those constraints
    therefore reports zero.

    Call after solving::

        from zen_garden.plugins.dclf.plugin import report_angle_spread
        report_angle_spread(optimization_setup)

    :param optimization_setup: the solved OptimizationSetup
    :param threshold: angle difference in radians above which to warn
    :return: dict with the largest spread, where it occurs, and how many
        (bus pair, time step) combinations exceed the threshold
    """
    model = optimization_setup.model
    if "theta" not in model.variables:
        logging.info(f"{LOG_PREFIX} no theta variable — nothing to report.")
        return {"max": 0.0, "edge": None, "time_step": None, "exceedances": 0}

    theta = model.variables["theta"].solution
    nodes_on_edges = optimization_setup.energy_system.set_nodes_on_edges
    angle_nodes = set(np.asarray(theta.coords["set_nodes"].data).tolist())

    # Recover the real KVL lines from the model: only edges carrying a
    # dclf_kvl_* row tie their endpoint angles together, and only within one
    # synchronous component are two angles relative to the same reference.
    # Rows suspended by the constraint mask (lines absent in a year) carry the
    # label -1; the angle difference is unconstrained there, so those time
    # steps are excluded per bus pair.
    pair_lines = {}  # (min(u, v), max(u, v)) -> (representative edge, active times)
    for name in model.constraints:
        if not name.startswith("dclf_kvl_"):
            continue
        constraint = model.constraints[name]
        active_rows = constraint.labels != -1
        for edge in np.asarray(constraint.coords["set_edges"].data).tolist():
            u, v = nodes_on_edges.get(edge, (None, None))
            if u is None or u not in angle_nodes or v not in angle_nodes:
                continue
            # An angle difference belongs to a pair of buses, not to a line, so
            # parallel corridors share one. Counting per line would make a
            # corridor of ten circuits look ten times as strained as it is.
            key = (min(u, v), max(u, v))
            edge_active = active_rows.sel({"set_edges": edge})
            if key in pair_lines:
                kept_edge, kept_active = pair_lines[key]
                pair_lines[key] = (kept_edge, kept_active | edge_active)
            else:
                pair_lines[key] = (edge, edge_active)

    if not pair_lines:
        logging.info(
            f"{LOG_PREFIX} no dclf_kvl_* constraint in the model (reloaded "
            f"without the plugin, or no KVL lines) — nothing to report."
        )
        return {"max": 0.0, "edge": None, "time_step": None, "exceedances": 0}

    node_component, _ = _components(
        [u for u, _v in pair_lines],
        [v for _u, v in pair_lines],
        sorted({n for pair in pair_lines for n in pair}),
    )

    worst, worst_edge, worst_time, exceedances = 0.0, None, None, 0
    for (u, v), (edge, active) in pair_lines.items():
        # Both endpoints of a KVL line share a synchronous component by
        # construction; the guard documents the invariant this diagnostic
        # relies on rather than filtering anything in practice.
        if node_component.get(u) != node_component.get(v):
            continue
        spread = np.abs(theta.sel(set_nodes=u) - theta.sel(set_nodes=v)).where(
            active, 0.0
        )
        difference = spread.data
        exceedances += int(np.count_nonzero(difference > threshold))
        local = float(np.max(difference, initial=0.0))
        if local > worst:
            worst = local
            worst_edge = edge
            worst_time = int(
                spread.coords["set_time_steps_operation"].data[int(np.argmax(difference))]
            )

    if worst > threshold:
        logging.warning(
            f"{LOG_PREFIX} voltage angle spread reaches {worst:.3f} rad "
            f"({np.degrees(worst):.1f} deg) on '{worst_edge}' at time step "
            f"{worst_time}; {exceedances} (bus pair, time step) combination(s) "
            f"exceed {threshold:g} rad. The DC linearisation is being used "
            f"outside the range where it is a good approximation."
        )
    else:
        logging.info(
            f"{LOG_PREFIX} largest voltage angle spread {worst:.3f} rad "
            f"({np.degrees(worst):.1f} deg), within the small-angle range."
        )
    return {
        "max": worst,
        "edge": worst_edge,
        "time_step": worst_time,
        "exceedances": exceedances,
    }
