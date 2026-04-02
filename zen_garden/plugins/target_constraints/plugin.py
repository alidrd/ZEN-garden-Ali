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
    "role": "source" | "destination" | "any"
  }

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


def _build_term(j, term, model, duration, nodes_on_edges):
    """
    Build a scalar linopy LinearExpression for one term:
      Σ_{filtered dims} variable * duration_weight
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

        elif isinstance(spec, dict) and spec.get("filter_type") == "edge_node_role": #NOTE: this is temporary/untested
            node = spec["node"]
            role = spec["role"]
            if role == "source":
                valid = [e for e in all_coords if nodes_on_edges[e][0] == node]
            elif role == "destination":
                valid = [e for e in all_coords if nodes_on_edges[e][1] == node]
            elif role == "any":
                valid = [e for e in all_coords if node in nodes_on_edges[e]]
            else:
                raise ValueError(f"[target_constraints] Unknown edge role '{role}'")
            filtered_coords[dim] = valid
            print(f"      '{dim}' (edge_node_role node='{node}' role='{role}'): {valid}")

        else:
            raise ValueError(
                f"[target_constraints] Unrecognised filter spec for "
                f"dim '{dim}': {spec!r}"
            )

    # ── STEP B: select filtered coordinates from the variable ─────────────────
    print(f"    [STEP B] Applying .sel() filters to variable:")
    var_filtered = var
    for dim, coords in filtered_coords.items():
        if not coords:
            raise ValueError(
                f"[target_constraints] Filter produced 0 coordinates for "
                f"dim '{dim}' in term {j}"
            )
        var_filtered = var_filtered.sel({dim: coords})

    print(f"    Filtered shape: {dict(zip(var_filtered.dims, var_filtered.shape))}")

    # ── STEP C: build scalar sum, weighting time steps by duration ────────────
    print(f"    [STEP C] Building weighted scalar sum:")
    has_time = "set_time_steps_operation" in var_filtered.dims

    if has_time:
        time_coords = filtered_coords["set_time_steps_operation"]
        print(f"      Time dimension present — weighting {len(time_coords)} step(s) by duration")
        duration_filtered = duration.sel(set_time_steps_operation=time_coords)
        expr = (var_filtered * duration_filtered).sum(var_filtered.dims)
        print(f"      Applied duration weights vectorially over dims {list(var_filtered.dims)}")
    else:
        print(f"      No time dimension — direct sum over all remaining dims")
        expr = var_filtered.sum()

    print(f"    Term {j} expression ready")
    return expr
