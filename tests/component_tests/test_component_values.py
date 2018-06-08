import unittest

from controller.controller import Controller
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_specifications import ComponentTypeSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.identifier.identifier_repository import IdentifierRepository


class TestComponentRepository(unittest.TestCase):

    def test_created_with_value_from_type(self):
        controller = Controller()
        controller.load_default_component_types()

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "DebugPrint"
        debug_print_type = controller.component_type_repository.get(component_type_spec)[0]
        new_value = debug_print_type.get_new_value()

        component_spec = ComponentSpecifications()
        component_spec.component_type_name = "DebugPrint"
        component = controller.component_repository.create(component_spec)
        component_value = component.value

        self.assertIsNotNone(component_value)
        self.assertEqual(new_value.__class__, component_value.__class__)