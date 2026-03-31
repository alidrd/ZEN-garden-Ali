"""
DC Load Flow (DCLF) plugin for ZEN-garden.

Adds Kirchhoff's voltage law (KVL) constraints on top of the standard
transport model. After each optimization construction, this plugin:
  1. Reads susceptance data from impedance.csv
  2. Adds voltage angle variables (theta) at each node/time step
  3. Adds flow equation constraints: P_ij = B_ij * (theta_i - theta_j)
  4. Adds reference bus constraint: theta_slack = 0
"""

import logging
import numpy as np
from zen_garden.events import Events, Event
from zen_garden.model.technology.transport_technology import TransportTechnology

# Populated by loader.py from config_dclf.json
config = {}


@Events.register(Event.after_optimization_construction)
def after_optimization_construction(optimization_setup, **kwargs):
    """DCLF hook: fires after the optimization model is fully constructed."""

    # ------------------------------------------------------------------ #
    # STEP 2: Read impedance from power_lines and compute susceptance
    # ------------------------------------------------------------------ #

    transport_techs = optimization_setup.get_all_elements(TransportTechnology)
    power_lines = next(
        (t for t in transport_techs if t.name == "power_lines"), None
    )

    if power_lines is None:
        logging.warning("DCLF plugin: no 'power_lines' technology found — skipping.")
        return

    impedance = power_lines.data_input.extract_input_data(
        file_name="impedance",
        index_sets=["set_edges"],
        unit_category={},
    )
    susceptance = 1.0 / impedance

    # ------------------------------------------------------------------ #
    # STEP 3: Inspect index sets and flow_transport variable
    # ------------------------------------------------------------------ #

    # Key sets we need for building DCLF constraints
    nodes      = list(optimization_setup.sets["set_nodes"])
    edges      = list(optimization_setup.sets["set_edges"])
    time_steps = list(optimization_setup.sets["set_time_steps_operation"])

    # Edge → (from_node, to_node) mapping, stored on energy_system
    nodes_on_edges = optimization_setup.energy_system.set_nodes_on_edges

    # The existing flow variable for transport technologies
    flow_transport = optimization_setup.model.variables["flow_transport"]

    # ------------------------------------------------------------------ #
    # STEP 3 TEST: print sets and variable info
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DCLF plugin — Step 3: index sets & flow_transport")
    print("=" * 60)
    print(f"\nnodes      : {nodes}")
    print(f"edges      : {edges}")
    print(f"time_steps : {time_steps}")
    print(f"\nnodes_on_edges (edge → (from, to)):")
    for edge, (frm, to) in nodes_on_edges.items():
        print(f"  {edge:10s} → from={frm}, to={to}")
    print(f"\nflow_transport dims   : {flow_transport.dims}")
    print(f"flow_transport coords : {dict(flow_transport.coords)}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # STEP 4: Add voltage angle variable theta[node, time_step]
    # ------------------------------------------------------------------ #
    # theta is unbounded (radians); the reference bus pin is added later.
    # We mirror ZEN-garden's dim naming convention so linopy aligns indices
    # consistently with the rest of the model.

    model = optimization_setup.model
    theta = model.add_variables(
        lower=-np.inf,
        upper=np.inf,
        coords=[nodes, time_steps],
        dims=["set_nodes", "set_time_steps_operation"],
        name="theta",
    )

    # ------------------------------------------------------------------ #
    # STEP 4 TEST: verify theta ended up in the model with correct shape
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DCLF plugin — Step 4: theta variable")
    print("=" * 60)
    print(f"theta dims   : {theta.dims}")
    print(f"theta coords : {dict(theta.coords)}")
    print(f"theta shape  : {theta.shape}")
    print(f"'theta' in model.variables: {'theta' in model.variables}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # STEP 4b: Allow bidirectional flow on power_lines
    # ------------------------------------------------------------------ #
    # flow_transport is bounded below by capacity.lower = 0, making it
    # unidirectional. DCLF requires signed flow: direction is determined
    # by angle differences, which can be positive OR negative.
    # Fix: set lower bound to -upper_bound for power_lines entries.

    ft = optimization_setup.model.variables["flow_transport"]
    pl_mask = ft.labels.loc["power_lines"].data != -1   # valid (non-padding) entries
    ft.lower.loc["power_lines"].data[pl_mask] = -ft.upper.loc["power_lines"].data[pl_mask]

    # ------------------------------------------------------------------ #
    # STEP 5: Add DCLF flow equality constraints
    # ------------------------------------------------------------------ #
    # ZEN-garden models each physical line as TWO directed edges
    # (e.g. AT-CH and CH-AT). Applying DCLF to both would force
    # flow_CH-AT = -flow_AT-CH, and ZEN-garden's energy balance would
    # count BOTH contributions, doubling the apparent power delivery.
    #
    # Fix: apply DCLF only to canonical edges (from_node < to_node),
    # and pin reverse edges to zero so they don't contribute to the
    # energy balance at all.

    canonical_edges = []   # one per physical line — DCLF applied here
    reverse_edges   = []   # paired duals — pinned to zero

    for edge in edges:
        fn, tn = nodes_on_edges[edge]
        if fn < tn:
            canonical_edges.append(edge)
        else:
            reverse_edges.append(edge)

    # DCLF equality on canonical edges
    first_edge_name = last_edge_name = None
    first_lhs       = last_lhs       = None

    for edge in canonical_edges:
        fn, tn = nodes_on_edges[edge]
        b = float(susceptance.loc[edge])

        flow_edge  = flow_transport.loc["power_lines", edge, :]
        theta_diff = theta.sel(set_nodes=fn) - theta.sel(set_nodes=tn)

        lhs = flow_edge - b * theta_diff
        model.add_constraints(lhs == 0, name=f"dclf_flow_{edge}")

        if first_edge_name is None:
            first_edge_name, first_lhs = edge, lhs
        last_edge_name, last_lhs = edge, lhs

    # Pin reverse edges to zero via bounds — prevents double-counting in energy balance.
    # Using bounds (lb = ub = 0) is strictly better than adding equality constraints:
    # the solver eliminates zero-bounded variables during presolve before the LP is
    # even handed to the simplex/interior-point method, so they add no rows to the
    # LP matrix and no computational cost at all.
    for edge in reverse_edges:
        mask = ft.labels.loc["power_lines", edge].data != -1
        ft.lower.loc["power_lines", edge].data[mask] = 0.0
        ft.upper.loc["power_lines", edge].data[mask] = 0.0

    # ------------------------------------------------------------------ #
    # STEP 5 TEST: verify constraints entered the model correctly
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DCLF plugin — Step 5: DCLF flow equality constraints")
    print("=" * 60)
    print(f"Canonical edges (DCLF applied) : {canonical_edges}")
    print(f"Reverse edges   (bounds→0, presolve-eliminated) : {reverse_edges}")
    print(f"\nConstraint names in model:")
    for name in model.constraints:
        if name.startswith("dclf_"):
            print(f"  {name}")
    if first_lhs is not None:
        print(f"\nFirst DCLF constraint — edge '{first_edge_name}' (t=0):")
        print(f"  {first_lhs.isel(set_time_steps_operation=0)}")
    if last_lhs is not None and last_edge_name != first_edge_name:
        print(f"\nLast DCLF constraint  — edge '{last_edge_name}' (t=0):")
        print(f"  {last_lhs.isel(set_time_steps_operation=0)}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # STEP 6: Reference bus constraint — theta_slack = 0 at all time steps
    # ------------------------------------------------------------------ #
    # Without this, angles are only determined up to a constant offset
    # (the LP is under-determined). Pinning one node fixes the gauge.
    # The slack node is read from the plugin config (default: "CH").

    slack_node = config.get("slack_node", "CH")
    theta_slack = theta.sel(set_nodes=slack_node)
    model.add_constraints(theta_slack == 0, name="dclf_ref_bus")

    # ------------------------------------------------------------------ #
    # STEP 6 TEST: confirm reference bus constraint is in the model
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DCLF plugin — Step 6: reference bus constraint")
    print("=" * 60)
    print(f"Slack node  : '{slack_node}'")
    print(f"'dclf_ref_bus' in model.constraints: {'dclf_ref_bus' in model.constraints}")
    print(f"\nExpression (t=0):")
    print(f"  {theta_slack.isel(set_time_steps_operation=0)}")
    print("=" * 60 + "\n")
    logging.info("DCLF plugin: all constraints added successfully.")
