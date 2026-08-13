"""
Target Constraints plugin for ZEN-garden.

Reads user-defined aggregation constraints from the plugin config and adds
them to the optimization model after construction.

Each constraint is a sum of one or more terms. Each term selects a model
variable, filters its dimensions, weights the time dimension by time step
duration (converting GW → GWh), and sums everything to a scalar. The scalar
is then compared against a fixed RHS.

Dimension filter spec for each dim:
  "all"             → include all coordinate values, sum over them
  ["v1", "v2", ...] → keep only listed values, sum over them
  {                 → special edge-node-role filter (for set_edges only)
    "filter_type": "edge_node_role",
    "node": "<node_name>",
    "role": "net_export" | "net_import" | "source" | "destination" | "any"
  }
  {                 → net flow over an explicit list of corridors
    "filter_type": "edge_net",
    "edges": ["DE-CH", ...]        // each minus its reverse edge
  }

Directed vs net flows
---------------------
ZEN-garden represents every corridor as two directed edges with non-negative
flows, and only their difference reaches the nodal energy balance. On a corridor
with no losses and no variable opex the pair is degenerate: the solver may return
a large flow in both directions whose difference is correct. Any constraint that
sums directed flows therefore counts circulating volume that does not physically
exist, and "exports from DE >= X" can be satisfied by flow that immediately
returns.

The netting roles remove this by construction, because circulation adds the same
amount to both directions of a corridor and cancels in the difference:

  net_export = Σ(edges where node is source) − Σ(edges where node is destination)
  net_import = the negative of that

The directional roles ("source", "destination", "any") are kept for gross
throughput targets and warn when used on flow variables, since they are only
meaningful when the model cannot circulate. Note that "any" double-counts every
corridor even without circulation, as it selects both directions.

Net flows are measured at the sending end; on a lossy corridor the importing node
receives slightly less.

Example config entry:
  {
    "comment": "Gas plant generation at AT >= 2 TWh",
    "terms": [
      {
        "variable": "flow_conversion_output",
        "dimensions": {
          "set_conversion_technologies": ["gas_power_plant"],
          "set_output_carriers":         ["electricity"],
          "set_nodes":                   ["AT"],
          "set_time_steps_operation":    "all"
        }
      }
    ],
    "sense": ">=",
    "rhs":   2.0,
    "unit":  "TWh"
  }
"""

import logging
import numpy as np
from zen_garden.events import Events, Event

# Populated by loader.py from the config JSON
config = {}

# Unit → GWh conversion (model internal energy unit is GWh)
_UNIT_TO_GWH = {
    "Wh":  1e-9,
    "MWh": 1e-3,
    "GWh": 1.0,
    "TWh": 1_000.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Event hook
# ─────────────────────────────────────────────────────────────────────────────

@Events.register(Event.after_optimization_construction)
def after_optimization_construction(optimization_setup, **kwargs):
    """Fires after the full linopy model is built. Adds target constraints."""

    constraints_cfg = config.get("target_constraints", [])
    if not constraints_cfg:
        logging.info("[target_constraints] No constraints defined — skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"[target_constraints] {len(constraints_cfg)} constraint(s) to add")
    print(f"{'=' * 60}")

    model      = optimization_setup.model
    parameters = optimization_setup.parameters
    nodes_on_edges = optimization_setup.energy_system.set_nodes_on_edges

    # ── PRINT 1: available variables ─────────────────────────────────────────
    print("\n[STEP 1] Variables present in model:")
    for vname in model.variables:
        v = model.variables[vname]
        print(f"  {vname}: dims={list(v.dims)}")

    # ── PRINT 2: time step durations ─────────────────────────────────────────
    duration = parameters.time_steps_operation_duration
    print("\n[STEP 2] Time step durations (hours):")
    for t, d in zip(
        duration.coords["set_time_steps_operation"].values,
        duration.values
    ):
        print(f"  t={t}: {float(d):.2f} h")

    # ── Add each constraint ───────────────────────────────────────────────────
    for i, cstr in enumerate(constraints_cfg):
        _add_constraint(i, cstr, model, duration, nodes_on_edges)

    print(f"\n{'=' * 60}")
    print(f"[target_constraints] All {len(constraints_cfg)} constraint(s) added.")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_constraint(i, cstr, model, duration, nodes_on_edges):
    """Parse one constraint entry and add it to the model."""
    comment  = cstr.get("comment", f"constraint_{i}")
    name_constraint = cstr.get("name", f"target_constraint_{i}")
    sense    = cstr["sense"]
    unit     = cstr.get("unit", "GWh")
    rhs_user = float(cstr["rhs"])
    rhs_gwh  = rhs_user * _UNIT_TO_GWH.get(unit, 1.0)

    print(f"\n[CONSTRAINT {i}] {comment}")
    print(f"  Sense : {sense}")
    print(f"  RHS   : {rhs_user} {unit}  →  {rhs_gwh} GWh")

    # Build scalar expression by summing all terms
    total_expr = None
    for j, term in enumerate(cstr["terms"]):
        expr = _build_term(j, term, model, duration, nodes_on_edges)
        total_expr = expr if total_expr is None else (total_expr + expr)
    
    print(f"\n  Created expression is: {total_expr}")

    # ── PRINT 3: confirm expression was built ────────────────────────────────
    print(f"  Total expression built → adding to model as 'target_constraint_{i}'")

    name = f"{name_constraint}_{i}"
    if name in model.constraints:
        raise ValueError(f"[target_constraints] Constraint name '{name}' already exists in model")
    
    if sense == "<=":
        model.add_constraints(total_expr <= rhs_gwh, name=name)
    elif sense == ">=":
        model.add_constraints(total_expr >= rhs_gwh, name=name)
    elif sense == "==":
        model.add_constraints(total_expr == rhs_gwh, name=name)
    else:
        raise ValueError(f"[target_constraints] Unknown sense '{sense}'")

    logging.info(f"[target_constraints] Added '{name}': {comment}")
    print(f"  ✓ '{name}' registered in model")


def _reverse_edge_map(nodes_on_edges):
    """Map every edge to the edge running the other way, where one exists."""
    edge_by_pair = {nodes_on_edges[e]: e for e in nodes_on_edges}
    return {
        e: edge_by_pair.get((v, u)) for e, (u, v) in nodes_on_edges.items()
    }


def _resolve_edge_role(spec, all_coords, nodes_on_edges, var_name):
    """Resolve an edge filter into (positive edges, negative edges).

    Netting roles return a non-empty negative list, whose flows are subtracted
    so that circulating volume cancels. Directional roles return an empty one.
    """
    filter_type = spec.get("filter_type")

    if filter_type == "edge_net":
        requested = spec["edges"]
        missing = [e for e in requested if e not in all_coords]
        if missing:
            raise ValueError(
                f"[target_constraints] Edges {missing} not found in '{var_name}'. "
                f"Available: {all_coords}"
            )
        reverse = _reverse_edge_map(nodes_on_edges)
        positive = list(requested)
        negative = [reverse[e] for e in requested if reverse.get(e) in all_coords]
        orphans = [e for e in requested if reverse.get(e) not in all_coords]
        if orphans:
            logging.warning(
                f"[target_constraints] Edges {orphans} have no reverse edge; their "
                f"flow is counted gross, not net."
            )
        return positive, negative

    node = spec["node"]
    role = spec["role"]
    outgoing = [e for e in all_coords if nodes_on_edges[e][0] == node]
    incoming = [e for e in all_coords if nodes_on_edges[e][1] == node]

    if role == "net_export":
        return outgoing, incoming
    if role == "net_import":
        return incoming, outgoing
    if role in ("source", "destination", "any"):
        if var_name.startswith("flow_"):
            logging.warning(
                f"[target_constraints] Role '{role}' sums directed flows of "
                f"'{var_name}'. On a corridor with no losses and no variable opex "
                f"the two directions are degenerate, so this total can include "
                f"circulating volume that never physically moves. Use "
                f"'net_export'/'net_import' unless a gross throughput target is "
                f"intended."
            )
        if role == "source":
            return outgoing, []
        if role == "destination":
            return incoming, []
        return [e for e in all_coords if node in nodes_on_edges[e]], []

    raise ValueError(f"[target_constraints] Unknown edge role '{role}'")


def _build_term(j, term, model, duration, nodes_on_edges):
    """
    Build a scalar linopy LinearExpression for one term:
      Σ_{filtered dims} variable * duration_weight

    Edge filters may be signed, in which case the term is the difference of two
    such sums (see the module docstring on directed vs net flows).
    """
    var_name = term["variable"]
    dims_cfg = term["dimensions"]

    print(f"\n  [TERM {j}] variable='{var_name}'")

    var = model.variables[var_name]
    print(f"    Variable dims : {list(var.dims)}")
    print(f"    Variable shape: {dict(zip(var.dims, var.shape))}")

    # ── STEP A: resolve valid coordinate list for every dim ───────────────────
    print(f"    [STEP A] Resolving dimension filters:")
    filtered_coords = {}
    edge_dim = None
    edge_positive, edge_negative = None, []

    for dim, spec in dims_cfg.items():
        all_coords = list(var.coords[dim].values)

        if spec == "all":
            filtered_coords[dim] = all_coords
            print(f"      '{dim}': ALL → {len(all_coords)} values: {all_coords}")

        elif isinstance(spec, list):
            valid = [c for c in all_coords if c in spec]
            missing = [c for c in spec if c not in all_coords]
            if missing:
                raise ValueError(
                    f"[target_constraints] Values {missing} not found in "
                    f"dim '{dim}' of '{var_name}'. Available: {all_coords}"
                )
            filtered_coords[dim] = valid
            print(f"      '{dim}': {valid}")

        elif isinstance(spec, dict) and spec.get("filter_type") in (
            "edge_node_role",
            "edge_net",
        ):
            if edge_dim is not None:
                raise ValueError(
                    f"[target_constraints] Term {j} has more than one edge filter"
                )
            edge_dim = dim
            edge_positive, edge_negative = _resolve_edge_role(
                spec, all_coords, nodes_on_edges, var_name
            )
            print(
                f"      '{dim}' ({spec.get('filter_type')}"
                f"{' role=' + spec['role'] if 'role' in spec else ''}): "
                f"+{edge_positive} -{edge_negative}"
            )

        else:
            raise ValueError(
                f"[target_constraints] Unrecognised filter spec for "
                f"dim '{dim}': {spec!r}"
            )

    # ── STEP B/C: select, weight by duration, sum to a scalar ─────────────────
    def _scalar_sum(edges):
        """Weighted scalar sum of the variable over one set of edges."""
        selection = dict(filtered_coords)
        if edge_dim is not None:
            selection[edge_dim] = edges
        var_filtered = var
        for dim, coords in selection.items():
            if not coords:
                raise ValueError(
                    f"[target_constraints] Filter produced 0 coordinates for "
                    f"dim '{dim}' in term {j}"
                )
            var_filtered = var_filtered.sel({dim: coords})
        if "set_time_steps_operation" in var_filtered.dims:
            duration_filtered = duration.sel(
                set_time_steps_operation=filtered_coords["set_time_steps_operation"]
            )
            return (var_filtered * duration_filtered).sum(var_filtered.dims)
        return var_filtered.sum()

    print(f"    [STEP B/C] Building weighted scalar sum:")
    expr = _scalar_sum(edge_positive)
    if edge_negative:
        print(f"      Subtracting {len(edge_negative)} opposing edge(s) → net flow")
        expr = expr - _scalar_sum(edge_negative)

    print(f"    Term {j} expression ready")
    return expr
