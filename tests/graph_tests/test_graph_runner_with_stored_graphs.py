import unittest

from controller.controller import Controller
from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestGraphRunnerWithStoredGraphs(unittest.TestCase):


    test_block_dir = "/home/michael/Projects/Mindblocks2.0/test_blocks/unit_tests/"

    def test_add_constants(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "add_constants_unit_test.xml"
        controller.load_block_file(test_block_file)

        output = controller.run_graphs()
        self.assertEqual(output[0][0], 8.15)

    def test_big_add(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "big_add_unit_test.xml"
        controller.load_block_file(test_block_file)

        output = controller.run_graphs()
        self.assertEqual(output[0][0], 244)

    def test_grid_add(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "grid_add_unit_test.xml"
        controller.load_block_file(test_block_file)

        output = controller.run_graphs()
        self.assertEqual(output[0][0], 40)

    def test_chain_add(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "chain_add_unit_test.xml"
        controller.load_block_file(test_block_file)

        output = controller.run_graphs()
        self.assertEqual(output[0][0], 20)