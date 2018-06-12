import unittest

from controller.controller import Controller
from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestTensorflowRunner(unittest.TestCase):

    test_block_dir = "/home/michael/Projects/Mindblocks2.0/test_blocks/unit_tests/"

    def test_compile_to_tensorflow(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "add_constants_tensorflow_unit_test.xml"
        controller.load_block_file(test_block_file)

        graph = controller.get_graphs()[0]
        compiled_copy = graph.get_compiled_copy()

        self.assertIsNotNone(compiled_copy)
        self.assertEqual(len(compiled_copy.vertices), 2)

        for vertex in graph.vertices:
            self.assertNotIn(vertex, compiled_copy.vertices)

        for edge in graph.edges:
            self.assertNotIn(edge, compiled_copy.edges)

    def test_tensorflow_add_constants(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "add_constants_tensorflow_unit_test.xml"
        controller.load_block_file(test_block_file)

        output = controller.run_graphs(compile=True)
        self.assertAlmostEqual(output[0][0], 8.15, places=6)

    def test_tensorflow_grid_add(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "grid_add_tensorflow_unit_test.xml"
        controller.load_block_file(test_block_file)

        output = controller.run_graphs(compile=True)
        self.assertAlmostEqual(output[0][0], 40, places=6)

    def test_tensorflow_grid_add_multiple_parts(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "grid_add_multiple_parts_tensorflow_unit_test.xml"
        controller.load_block_file(test_block_file)

        output = controller.run_graphs(compile=True)
        self.assertAlmostEqual(output[0][0], 40, places=6)