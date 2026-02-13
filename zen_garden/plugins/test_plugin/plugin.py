"""
Example plugin demonstrating the ZEN-garden plugin contract.

This plugin serves as a reference implementation for developers creating
new plugins.
"""
from zen_garden.plugins.base_plugin import BasePlugin
from zen_garden.plugin_manager import Hook
from zen_garden.utils import setup_logger

import logging

setup_logger()


class Plugin(BasePlugin):
    """
    Example plugin that logs messages at optimization lifecycle hooks.

    Attributes
    ----------
    name : str
        Human-readable plugin name.
    hooks : set
        Hooks this plugin subscribes to.
    config_template : dict
        Example configuration schema.
    """

    name = "Test Plugin"
    hooks = {
        Hook.BEFORE_OPTIMIZATION_CONSTRUCTION,
        Hook.AFTER_OPTIMIZATION_CONSTRUCTION,
    }
    config_template = {"config1": "TestString"}

    def before_optimization_construction(self, optimization_setup):
        """
        Called before optimization model is constructed.

        Parameters
        ----------
        optimization_setup : object
            The optimization setup context.
        """
        logging.info(f"Plugin {self.name} can perform actions before optimization construction.")

    def after_optimization_construction(self, optimization_setup, model_instance):
        """
        Called after optimization model is constructed.

        Parameters
        ----------
        optimization_setup : object
            The optimization setup context.
        model_instance : object
            The constructed optimization model.
        """
        logging.info(f"Plugin {self.name} can perform actions after optimization construction.")
