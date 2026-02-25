from zen_garden.events import Events, Event

config = {}
"""
This config dictionary will be filled by 
plugins.loader.import_selection_of_plugins()
"""

Events.register(Event.before_optimization_construction)
def do_something_before_optimization_construction(*args, **kwargs):
    pass
