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


    def test_config_is_replaced_not_merged_between_runs(self):
        # Arrange
        from tests.unit_tests.fake_plugin import plugin
        import_selection_of_plugins(
            {"fake_plugin": {"stale_parameter": "stale_value"}},
            source_package="tests.unit_tests"
        )
        held_reference = plugin.config

        # Act
        import_selection_of_plugins(
            {"fake_plugin": {"fresh_parameter": "fresh_value"}},
            source_package="tests.unit_tests"
        )

        # Assert
        assert plugin.config == {"fresh_parameter": "fresh_value"}
        assert held_reference == {"fresh_parameter": "fresh_value"}


    def test_previously_imported_plugin_is_disabled_when_not_selected(self):
        # Arrange
        from tests.unit_tests.fake_plugin import plugin
        import_selection_of_plugins(
            {"fake_plugin": {"any_parameter": "any_value"}},
            source_package="tests.unit_tests"
        )
        assert plugin.enabled is True

        # Act: a later run in the same process selects no plugins
        result = import_selection_of_plugins(
            {},
            source_package="tests.unit_tests"
        )

        # Assert
        assert result == {}
        assert plugin.enabled is False
        assert plugin.config == {}