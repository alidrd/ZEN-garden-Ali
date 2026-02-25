from zen_garden.plugins.loader import import_selection_of_plugins

class TestPluginsLoader():


    def test_import_selected_plugin_import_corresponding_module(self):
        # Arrange
        plugins = {"fake_plugin": {}}

        # Act
        result = import_selection_of_plugins(
            plugins,
            source_package="tests.unit_tests"
        )
        from tests.unit_tests.fake_plugin import plugin

        # Assert
        assert result["fake_plugin"] == plugin


    def test_pass_config_to_selected_plugins(self):
        # Arrange
        plugins = {"fake_plugin": {"any_parameter": "any_value"}}

        # Act
        import_selection_of_plugins(
            plugins,
            source_package="tests.unit_tests"
        )
        from tests.unit_tests.fake_plugin import plugin

        # Assert
        assert plugin.config == plugins["fake_plugin"] 