import unittest

from controller.component_type_loader.component_type_loader import ComponentTypeLoader
from controller.controller import Controller


class TestComponentTypeLoader(unittest.TestCase):

    def test_loads_folder(self):
        component_type_loader = ComponentTypeLoader()
        component_types = component_type_loader.load_component_type_folder("/home/michael/Projects/Mindblocks2.0/component_types")
        names = [c.name for c in component_types]

        self.assertGreater(len(component_types), 0)
        self.assertIn("Debug Printer", names)

    def test_loads_subfolders(self):
        pass

    def test_loaded_types_added_to_repository(self):
        pass

