"""
Plugin manager for ZEN-garden

This module provides the :class:`PluginManager` which discovers, loads,
and emits hooks to plugins, as well as the :class:`Hook` enum defining
available extension points.
"""

import logging
import pkgutil
from enum import Enum
from importlib import import_module
from typing import List

from zen_garden.plugins.base_plugin import BasePlugin
from .utils import setup_logger

setup_logger()

class Hook(Enum):

    BEFORE_OPTIMIZATION_CONSTRUCTION = "before_optimization_construction"
    AFTER_OPTIMIZATION_CONSTRUCTION = "after_optimization_construction"


class PluginManager:
    """
    Plugin manager

    Responsibilities (minimal):
    - Load plugins listed in a config
    - Auto-discover plugins under the main ``plugins`` package
    - Emit hooks to loaded plugins
    """

    def __init__(self, base_package: str = "zen_garden.plugins"):
        self.base_package = base_package
        self._plugins: List[BasePlugin] = []

    def discover_plugins(self) -> List[str]:
        """
        Auto-discover plugin modules.

        This implementation looks for plugin modules directly under the
        configured ``base_package`` (``zen_garden.plugins``).

        Returns
        -------
        List[str]
            A list of import paths for modules that expose a ``Plugin``
            class.
        """
        discovered: List[str] = []

        # First, try the base package itself and discover top-level plugin
        # modules/packages directly under it.
        try:
            base_package = import_module(self.base_package)
        except ImportError:
            logging.exception("Plugin package %s not found", self.base_package)
            return discovered

        # Iterate over modules/packages in the base package path
        for finder, name, ispkg in pkgutil.iter_modules(base_package.__path__):
            if name.startswith("_"):
                continue

            module_path = f"{self.base_package}.{name}"
            try:
                mod = import_module(module_path)
                if hasattr(mod, "Plugin"):
                    discovered.append(module_path)
                    logging.info("Discovered plugin %s", module_path)

            except Exception:
                logging.exception("Failed importing discovered plugin %s", module_path)

        return discovered


    def register(self, plugin_ids: dict):
        """
        Register plugins based on the provided list.

        The config may be a list of identifiers
        """
        if plugin_ids is None:
            logging.debug("No plugin config provided; skipping")
            return

        for plugin_id, plugin_config in plugin_ids.items():
            plugin_config = plugin_config["config"]
            module_path = f"{self.base_package}.{plugin_id}"
            if not module_path:
                logging.exception("Could not resolve plugin identifier: %s", plugin_id)
                continue
            try:
                plugin_module = import_module(module_path)
                plugin = getattr(plugin_module, "Plugin", None)
                plugin_instance = plugin(plugin_config)
                self._plugins.append(plugin_instance)
                logging.info("Loaded plugin %s", getattr(plugin_instance, "name", module_path))
            except Exception:
                logging.exception("Failed loading plugin %s", module_path)

    def emit(self, hook: Hook, **kwargs):
        """
        Call hook on all plugins. One can pass additional arguments via kwargs.
        """
        hook_name = hook.value
        for plugin_instance in list(self._plugins):
            plugin_hook = getattr(plugin_instance, hook_name, None)
            plugin_hook(**kwargs)

    def get_plugins(self):
        return list(self._plugins)

    def print_plugins(self):
        available_plugins = self.discover_plugins()

        for plugin in available_plugins:
            print(plugin)

    def deregister_all(self):
        """
        Deregister all plugins.
        """
        self._plugins.clear()

