from zen_garden.plugin_manager import PluginManager


def test_plugin_manager_plugin_discovery():
    """Verify PluginManager discovers expected plugins."""
    pm = PluginManager()
    discovered = set(pm.discover_plugins())

    expected = {"zen_garden.plugins.test_plugin"}
    assert discovered == expected
