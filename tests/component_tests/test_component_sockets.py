import unittest

from controller.controller import Controller
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_specifications import ComponentTypeSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.identifier.identifier_repository import IdentifierRepository


class TestComponentSockets(unittest.TestCase):

    def test_created_from_type(self):
        controller = Controller()
        controller.load_default_component_types()

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "DebugPrint"
        debug_print_type = controller.component_type_repository.get(component_type_spec)[0]
        n_in_sockets = debug_print_type.in_degree()
        n_out_sockets = debug_print_type.out_degree()

        component_spec = ComponentSpecifications()
        component_spec.component_type_name = "DebugPrint"
        component = controller.component_repository.create(component_spec)


        self.assertIsNotNone(component.in_sockets)
        self.assertIsNotNone(component.out_sockets)
        self.assertEqual(len(component.in_sockets), n_in_sockets)
        self.assertEqual(len(component.out_sockets), n_out_sockets)