import unittest

from controller.block_loader.block_loader import BlockLoader
from controller.controller import Controller
from helpers.xml.xml_helper import XmlHelper
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.component.component_specifications import ComponentSpecifications
from repository.graph.graph_specifications import GraphSpecifications


class TestBlockLoader(unittest.TestCase):

    test_block_dir = "/home/michael/Projects/Mindblocks2.0/test_blocks/"

    def test_loads_canvases(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "debug_print.xml"
        controller.load_block_file(test_block_file)

        self.assertGreater(len(controller.canvas_repository.elements), 0)
        self.assertEqual(len(controller.canvas_repository.elements), 2)

    def test_loads_canvases_with_names(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "debug_print.xml"
        controller.load_block_file(test_block_file)

        specs = CanvasSpecifications()
        specs.name="main"
        canvases_1 = controller.canvas_repository.get(specs)
        self.assertEqual(len(canvases_1), 1)

        specs.name="secondary"
        canvases_2 = controller.canvas_repository.get(specs)
        self.assertEqual(len(canvases_2), 1)

        self.assertNotEqual(canvases_1[0].identifier, canvases_2[0].identifier)

    def test_loads_components(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "debug_print.xml"
        controller.load_block_file(test_block_file)

        self.assertGreater(len(controller.component_repository.elements), 0)
        self.assertEqual(len(controller.component_repository.elements), 2)

    def test_loads_components_and_assigns_correct_canvas(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "debug_print.xml"
        controller.load_block_file(test_block_file)

        component_specification = ComponentSpecifications()
        component_specification.name = "main_print"
        components = controller.component_repository.get(component_specification)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].canvas_name, "main")

        component_specification.name = "secondary_print"
        components = controller.component_repository.get(component_specification)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].canvas_name, "secondary")

    def test_loads_components_and_assigns_correct_types(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "debug_print.xml"
        controller.load_block_file(test_block_file)

        component_specification = ComponentSpecifications()
        component_specification.name = "main_print"
        components = controller.component_repository.get(component_specification)

        self.assertIsNotNone(components[0].component_type.name)
        self.assertEqual(components[0].component_type.name, "DebugPrint")

        component_specification.name = "secondary_print"
        components = controller.component_repository.get(component_specification)

        self.assertIsNotNone(components[0].component_type.name)
        self.assertEqual(components[0].component_type.name, "DebugPrint")

    def test_loads_values(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "debug_print.xml"
        controller.load_block_file(test_block_file)

        component_specification = ComponentSpecifications()
        component_specification.name = "main_print"
        component = controller.component_repository.get(component_specification)[0]

        self.assertIsNotNone(component.value)
        self.assertIsNotNone(component.value.text)
        self.assertEqual(component.value.text, "test message")

    def test_loads_edges(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "print_constant.xml"
        controller.load_block_file(test_block_file)

        component_specification = ComponentSpecifications()
        component_specification.name = "constant"
        component = controller.component_repository.get(component_specification)[0]

        self.assertIsNotNone(component.out_sockets)
        self.assertEqual(len(component.out_sockets), 1)

        component_specification.name = "printer"
        component_2 = controller.component_repository.get(component_specification)[0]

        self.assertIsNotNone(component_2.in_sockets)
        self.assertEqual(len(component_2.in_sockets), 1)

        graph_spec = GraphSpecifications()
        graph_spec.identifier = component.graph_id
        graph = controller.graph_repository.get(graph_spec)[0]

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0], component.out_sockets[0][0])
        self.assertEqual(graph.edges[0], component_2.in_sockets[0])

    def test_loads_many_edges(self):
        controller = Controller()
        controller.load_default_component_types()
        test_block_file = self.test_block_dir + "add_constants.xml"
        controller.load_block_file(test_block_file)

        component_specification = ComponentSpecifications()
        component_specification.name = "constant_1"
        constant_1 = controller.component_repository.get(component_specification)[0]

        self.assertIsNotNone(constant_1.out_sockets)
        self.assertEqual(len(constant_1.out_sockets), 1)

        component_specification = ComponentSpecifications()
        component_specification.name = "constant_2"
        constant_2 = controller.component_repository.get(component_specification)[0]

        self.assertIsNotNone(constant_2.out_sockets)
        self.assertEqual(len(constant_2.out_sockets), 1)

        component_specification = ComponentSpecifications()
        component_specification.name = "adder"
        adder = controller.component_repository.get(component_specification)[0]

        self.assertIsNotNone(adder.in_sockets)
        self.assertEqual(len(adder.in_sockets), 2)
        self.assertIsNotNone(adder.out_sockets)
        self.assertEqual(len(adder.out_sockets), 1)

        component_specification = ComponentSpecifications()
        component_specification.name = "printer"
        printer = controller.component_repository.get(component_specification)[0]

        self.assertIsNotNone(printer.in_sockets)
        self.assertEqual(len(printer.in_sockets), 1)

        graph_spec = GraphSpecifications()
        graph_spec.identifier = printer.graph_id
        graph = controller.graph_repository.get(graph_spec)[0]

        self.assertEqual(len(graph.edges), 3)
        self.assertIn(constant_1.out_sockets[0][0], graph.edges)
        self.assertIn(constant_2.out_sockets[0][0], graph.edges)
        self.assertIn(adder.out_sockets[0][0], graph.edges)

        self.assertEqual(constant_1.out_sockets[0][0], adder.in_sockets[0])
        self.assertEqual(constant_2.out_sockets[0][0], adder.in_sockets[1])
        self.assertEqual(adder.out_sockets[0][0], printer.in_sockets[0])