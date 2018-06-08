import unittest

from controller.component_type_loader.component_type_loader import ComponentTypeLoader
from controller.controller import Controller


class TestComponentTypeLoader(unittest.TestCase):

    default_component_type_folder = "C:/Users/micha/OneDrive/Dokumenter/Projects/Mindblocks2.0/component_types"

    def test_loads_folder(self):
        component_type_loader = ComponentTypeLoader()
        component_types = component_type_loader.load_component_type_folder(self.default_component_type_folder)
        names = [c.name for c in component_types]

        self.assertGreater(len(component_types), 0)
        self.assertIn("DebugPrint", names)

    def test_loads_subfolders(self):
        component_type_loader = ComponentTypeLoader()
        component_types = component_type_loader.load_component_type_folder(self.default_component_type_folder)

        names = [c.name for c in component_types]

        self.assertGreater(len(component_types), 1)
        self.assertIn("Add", names)

    def test_loaded_types_added_to_repository(self):
        controller = Controller()
        component_types = controller.load_component_types(self.default_component_type_folder)

        self.assertGreater(len(controller.component_type_repository.elements), 0)
        self.assertEqual(len(controller.component_type_repository.elements), len(component_types))

    def test_controller_loads_default_components(self):
        controller = Controller()
        controller.load_default_component_types()

        self.assertGreater(len(controller.component_type_repository.elements), 0)
        self.assertIn("DebugPrint", [e.name for e in controller.component_type_repository.elements])

