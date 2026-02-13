from zen_garden.plugin_manager import PluginManager


def test_plugin_manager():
    """Verify PluginManager discovers the test_plugin package."""
    pm = PluginManager()
    discovered = set(pm.discover_plugins())
    pm.print_plugins()

    expected = {"zen_garden.plugins.test_plugin"}
    assert discovered == expected
