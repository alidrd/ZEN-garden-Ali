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

    # Find the power_lines transport technology object
    transport_techs = optimization_setup.get_all_elements(TransportTechnology)
    power_lines = next(
        (t for t in transport_techs if t.name == "power_lines"), None
    )

    if power_lines is None:
        logging.warning("DCLF plugin: no 'power_lines' technology found — skipping.")
        return

    # Read impedance.csv using ZEN-garden's own data machinery
    # (handles unit conversion, defaults, index alignment automatically)
    impedance = power_lines.data_input.extract_input_data(
        file_name="impedance",
        index_sets=["set_edges"],
        unit_category={},
    )
    susceptance = 1.0 / impedance

    # ------------------------------------------------------------------ #
    # STEP 2 TEST: print impedance and susceptance to verify
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DCLF plugin — Step 2: impedance & susceptance")
    print("=" * 60)
    print(f"\nimpedance (from CSV):\n{impedance}")
    print(f"\nsusceptance (1/impedance):\n{susceptance}")
    print("=" * 60 + "\n")
