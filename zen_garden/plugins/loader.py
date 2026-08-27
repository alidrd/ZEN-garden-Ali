"""
This module only functionality is to import the modules from the plugins package
and update their config dictionaries based on the user specifications from
config.json

Because ``@Events.register`` hooks are registered at module import and
``importlib`` caches modules, a plugin imported once stays hooked for the
lifetime of the process. To keep runs isolated, the loader tracks every plugin
module it has imported in this process: plugins selected for the current run
get their ``config`` replaced (not merged) and ``enabled = True``, while every
previously imported plugin that is *not* selected gets its ``config`` cleared
and ``enabled = False`` so its hook returns immediately.
"""

import importlib
from types import ModuleType

# Every plugin module imported in this process, keyed by plugin name.
_imported: dict[str, ModuleType] = {}

def import_selection_of_plugins(
    plugins_config: dict[str, dict],
    source_package: str = "zen_garden.plugins"
) -> dict[str, ModuleType]:
    output = {}
    for plugin, config in plugins_config.items():
        module = importlib.import_module(
            name=f"{source_package}.{plugin}.plugin"
        )
        # Replace, not merge — mutate in place so references held by the
        # plugin module keep seeing the current run's config.
        module.config.clear()
        module.config.update(config)
        module.enabled = True
        _imported[plugin] = module
        output[plugin] = module
    for plugin, module in _imported.items():
        if plugin not in plugins_config:
            module.config.clear()
            module.enabled = False
    return output
