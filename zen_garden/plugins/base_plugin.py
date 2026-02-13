"""
Base plugin implementation and lightweight plugin contract.

This module defines a minimal plugin base class
that other plugin implementations should inherit from. The plugin
infrastructure in `zen_garden` performs two roles:

- provide a small, stable contract for plugins (``BasePlugin``), and
- allow the :mod:`zen_garden.core.plugin_manager` to discover, load
  and emit hooks to plugins.

Example
-------
A minimal plugin implementation::

    # my_plugin.py
    from zen_garden.plugins.base_plugin import BasePlugin

    class Plugin(BasePlugin):
        name = "my_plugin"
        hooks = {"test_hook"}

        def test_hook(self, **kwargs):
            # called by PluginManager.emit(Hook.TEST_HOOK)
            return f"handled: {kwargs}"

"""

class BasePlugin:
    """
    Minimal base plugin (all plugins should subclass this class).

    Attributes
    ----------
    name : str
        Human-readable plugin name.

    hooks : set[str]
        A set containing the names of hook methods the plugin implements.
        The plugin manager uses hook names (strings) to look up callables
        on plugin instances when emitting events.

    Instance attributes
    -------------------
    config : dict
        Plugin-specific configuration passed in by the caller (usually
        loaded from the project configuration)

    Notes
    -----
    The base class intentionally implements minimal behavior. Subclasses
    are responsible for implementing hook methods.
    """

    name = "base"

    hooks = set()

    def __init__(self, config: dict | None = None):
        """
        Initialize the plugin.

        Parameters
        ----------
        config : dict | None
            Plugin-specific configuration passed in by the caller (usually
            loaded from the project configuration)
        """
        self.config = config or {}

    def activate(self) -> None:
        """
        Activate the plugin.

        This method is called once by the :class:`~zen_garden.core.plugin_manager.PluginManager`
        after the plugin has been instantiated.

        The default implementation is a no-op and returns ``None``.

        Returns
        -------
        None

        """
        return None
