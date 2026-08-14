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
opex or a carbon intensity is non-zero. The plugin verifies all three are zero
and refuses to use this mode otherwise.

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

Configuration
-------------
Declared under ``plugins`` in the run config::

    "plugins": {
      "dclf": {
        "kvl_nodes":           ["DE*"],   // names or fnmatch patterns; default all
        "technologies":        ["power_lines"], // default: techs with impedance
        "impedance_file":      "impedance",
        "flow_representation": "auto",    // "auto" | "signed" | "net"
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
)

# Populated by loader.py from the run config
config = {}

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


def _read_susceptance(tech, impedance_file):
    """Read impedances for one technology and invert them to susceptances.

    :param tech: TransportTechnology object
    :param impedance_file: name of the impedance input file
    :return: pandas Series of susceptance indexed by edge
    """
    impedance = tech.data_input.extract_input_data(
        file_name=impedance_file, index_sets=["set_edges"], unit_category={}
    )
    # extract_input_data returns a single-level MultiIndex; flatten it so that
    # .loc[edge] yields a scalar rather than a one-element Series
    if impedance.index.nlevels == 1 and isinstance(impedance.index, pd.MultiIndex):
        impedance.index = impedance.index.get_level_values(0)
    invalid = impedance[~np.isfinite(impedance) | (impedance <= 0)]
    if len(invalid) > 0:
        raise DCPFDataError(
            f"{LOG_PREFIX} technology '{tech.name}' has non-positive or "
            f"non-finite impedance on edges {sorted(invalid.index)}. "
            f"Every edge inside the KVL region needs a positive, finite impedance."
        )
    return 1.0 / impedance


def _build_lines(
    tech, susceptance, kvl_nodes, nodes_on_edges, valid_edges, reverse_by_edge
):
    """Pair the directed edges of one technology into undirected KVL lines.

    An edge qualifies when both of its endpoints are in the KVL node set. Its
    reverse partner must exist, because the constraint is written on the net
    flow of the pair. Parallel corridors between the same two nodes are kept as
    separate lines, each paired with its own reverse edge and carrying its own
    susceptance — parallel susceptances then add up through the shared voltage
    angles, which is what makes an aggregated corridor behave correctly without
    a blended equivalent impedance.

    :param tech: TransportTechnology object
    :param susceptance: susceptance per edge
    :param kvl_nodes: set of nodes inside the KVL region
    :param nodes_on_edges: dict edge -> (node_from, node_to)
    :param valid_edges: edges for which this technology has a flow variable
    :param reverse_by_edge: dict edge -> reverse edge (or None)
    :return: dict of parallel lists describing the lines
    """
    kvl_nodes = set(kvl_nodes)

    line_ids, fwd, rev, from_nodes, to_nodes, b_values = [], [], [], [], [], []
    missing_reverse, missing_variable, self_loops = [], [], []

    for edge, (u, v) in nodes_on_edges.items():
        if u not in kvl_nodes or v not in kvl_nodes:
            continue  # boundary or fully external edge -> stays a transport edge
        if u == v:
            self_loops.append(edge)
            continue
        if u > v:
            continue  # handled from its canonical partner
        reverse = reverse_by_edge.get(edge)
        if reverse is None:
            missing_reverse.append(edge)
            continue
        if edge not in valid_edges or reverse not in valid_edges:
            missing_variable.append(edge)
            continue
        line_ids.append(edge)
        fwd.append(edge)
        rev.append(reverse)
        from_nodes.append(u)
        to_nodes.append(v)
        b_values.append(float(susceptance.loc[edge]))

    if self_loops:
        logging.warning(
            f"{LOG_PREFIX} technology '{tech.name}': ignoring {len(self_loops)} "
            f"self-loop edge(s) inside the KVL region: {sorted(self_loops)[:5]}"
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
    return {
        "line_ids": line_ids,
        "fwd": fwd,
        "rev": rev,
        "from_nodes": from_nodes,
        "to_nodes": to_nodes,
        "b": b_values,
    }


def _lossless_and_free(optimization_setup, tech_name, edges):
    """Check that a technology carries no loss, no variable opex, no emissions.

    These are exactly the quantities that are tied by equality to a non-negative
    variable times the flow, and therefore forbid a negative flow.

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

    ok, offenders = _lossless_and_free(optimization_setup, tech_name, edges)
    if requested == "signed" and not ok:
        raise DCPFDataError(
            f"{LOG_PREFIX} flow_representation='signed' requires a lossless, "
            f"cost-free line, but technology '{tech_name}' has non-zero "
            f"{offenders} on KVL edges. These are tied by equality to a "
            f"non-negative variable times the flow, so a signed flow would make "
            f"the model infeasible. Set them to zero or use "
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


def _warn_on_bypass_paths(lines_by_tech, kvl_nodes, nodes_on_edges, node_component):
    """Warn about controllable corridors parallel to the KVL subgraph.

    An edge of another technology whose endpoints sit in the same synchronous
    component lets flow route around the impedance constraint. That is legitimate
    for a genuine HVDC embedded in an AC grid, and a modelling error otherwise,
    so it is reported rather than rejected.

    :param lines_by_tech: dict tech name -> line description
    :param kvl_nodes: set of nodes inside the KVL region
    :param nodes_on_edges: dict edge -> (node_from, node_to)
    :param node_component: dict node -> component id
    """
    constrained = {e for d in lines_by_tech.values() for e in d["fwd"] + d["rev"]}
    bypass = [
        edge
        for edge, (u, v) in nodes_on_edges.items()
        if u < v
        and u in kvl_nodes
        and v in kvl_nodes
        and edge not in constrained
        and node_component.get(u) is not None
        and node_component.get(u) == node_component.get(v)
    ]
    if bypass:
        logging.warning(
            f"{LOG_PREFIX} {len(bypass)} corridor(s) inside a synchronous component "
            f"are not KVL-constrained and can carry flow around the impedance "
            f"constraint: {sorted(bypass)[:10]}"
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

    Zero-width bounds are strictly better than an equality constraint: the
    solver eliminates the reverse variables in presolve, so they add no rows.

    :param flow: the flow_transport variable
    :param tech_name: name of the transport technology
    :param lines: line description produced by :func:`_build_lines`
    """
    for direction, edges in (("forward", lines["fwd"]), ("reverse", lines["rev"])):
        sel = {"set_transport_technologies": tech_name, "set_edges": edges}
        valid = flow.labels.loc[sel] != -1
        if direction == "forward":
            flow.lower.loc[sel] = xr.where(
                valid, -flow.upper.loc[sel], flow.lower.loc[sel]
            )
        else:
            flow.lower.loc[sel] = xr.where(valid, 0.0, flow.lower.loc[sel])
            flow.upper.loc[sel] = xr.where(valid, 0.0, flow.upper.loc[sel])


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


@Events.register(Event.after_optimization_construction)
def after_optimization_construction(optimization_setup, **kwargs):
    """Add DC power flow constraints to the constructed model.

    :param optimization_setup: the OptimizationSetup the plugin operates on
    """
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
    for tech in techs:
        susceptance = _read_susceptance(tech, impedance_file)
        lines = _build_lines(
            tech, susceptance, kvl_nodes, nodes_on_edges, valid_edges, reverse_by_edge
        )
        if lines["line_ids"]:
            lines_by_tech[tech.name] = lines

    if not lines_by_tech:
        logging.warning(
            f"{LOG_PREFIX} the KVL node set {kvl_nodes} contains no internal line "
            f"— no constraints added."
        )
        return

    # ---- synchronous components, over the union of all KVL lines -----------
    all_from = [u for d in lines_by_tech.values() for u in d["from_nodes"]]
    all_to = [v for d in lines_by_tech.values() for v in d["to_nodes"]]
    node_component, used_components = _components(all_from, all_to, nodes)
    slack_nodes = _resolve_slack_nodes(node_component, used_components)
    _warn_on_bypass_paths(lines_by_tech, set(kvl_nodes), nodes_on_edges, node_component)

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

        if representation == "signed":
            # The reverse edge is removed from the problem and the forward edge
            # carries the signed flow, so the solution is unique.
            _set_signed_bounds(flow, tech_name, lines)
            lhs = flow_tech.sel({"set_edges": fwd_idx}) - susceptance * angle_difference
            model.add_constraints(lhs == 0, name=f"dclf_kvl_{tech_name}")
            # ZEN-garden's stock capacity constraint is one-sided
            # (flow <= max_load * capacity); a signed flow needs the mirror image.
            term_capacity = (
                optimization_setup.parameters.max_load.loc[tech_name, line_ids, :]
                * model.variables["capacity"].loc[
                    tech_name, "power", line_ids, time_step_year
                ]
            ).rename({"set_location": "set_edges"})
            model.add_constraints(
                flow_tech.sel({"set_edges": fwd_idx}) + term_capacity >= 0,
                name=f"dclf_capacity_reverse_{tech_name}",
            )
        else:
            net_flow = flow_tech.sel({"set_edges": fwd_idx}) - flow_tech.sel(
                {"set_edges": rev_idx}
            )
            lhs = net_flow - susceptance * angle_difference
            model.add_constraints(lhs == 0, name=f"dclf_kvl_{tech_name}")

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
