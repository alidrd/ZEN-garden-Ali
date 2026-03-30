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
