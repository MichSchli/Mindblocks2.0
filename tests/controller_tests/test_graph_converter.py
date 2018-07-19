import unittest

from controller.block_loader.block_loader import BlockLoader
from controller.block_loader.canvas_loader import CanvasLoader
from controller.block_loader.component_loader import ComponentLoader
from controller.block_loader.edge_loader import EdgeLoader
from controller.component_type_loader.component_type_loader import ComponentTypeLoader
from controller.graph_converter.graph_converter import GraphConverter
from helpers.files.FilepathHandler import FilepathHandler
from helpers.xml.xml_helper import XmlHelper
from repository.canvas_repository.canvas_repository import CanvasRepository
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.identifier.identifier_repository import IdentifierRepository

class TestGraphConverter(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.canvas_repository = CanvasRepository(self.identifier_repository)
        self.graph_repository = GraphRepository(self.identifier_repository)
        self.component_repository = CreationComponentRepository(self.identifier_repository,
                                                                self.type_repository,
                                                                self.canvas_repository,
                                                                self.graph_repository)

        self.filepath_handler = FilepathHandler()
        self.component_type_loader = ComponentTypeLoader(self.filepath_handler, self.type_repository)
        self.component_type_loader.load_default_folder()

        self.xml_helper = XmlHelper()
        self.component_loader = ComponentLoader(self.xml_helper, self.component_repository)
        self.edge_loader = EdgeLoader(self.xml_helper, self.graph_repository, self.component_repository)
        self.canvas_loader = CanvasLoader(self.xml_helper, self.component_loader, self.edge_loader,
                                          self.canvas_repository)
        self.block_loader = BlockLoader(self.xml_helper, self.canvas_loader)

        self.graph_converter = GraphConverter()

    def testInitializingValues(self):
        filename = "add_constants.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        value_dictionary = self.graph_converter.build_value_dictionary(runs)

        self.assertIsNotNone(value_dictionary)
        self.assertEqual(3, len(value_dictionary))

        keys = list(value_dictionary.keys())

        for component in list(self.component_repository.elements.values()):
            self.assertIn(component.identifier, keys)
            self.assertIsNotNone(value_dictionary[component.identifier])

    def testExcludesIrrelevantPartsFromValues(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        value_dictionary = self.graph_converter.build_value_dictionary(runs)

        self.assertIsNotNone(value_dictionary)
        self.assertEqual(3, len(value_dictionary))

        keys = list(value_dictionary.keys())

        for component in list(self.component_repository.elements.values()):
            if component.name != "adder_2":
                self.assertIn(component.identifier, keys)
                self.assertIsNotNone(value_dictionary[component.identifier])

    def testCreatesExecutionGraphs(self):
        filename = "add_constants.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())

    def testExcludesIrrelevantPartsFromExecution(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())

    def testExecutesMultipleRuns(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        component_spec.name = "adder_2"
        adder_2 = self.component_repository.get(component_spec)[0]
        target_socket_2 = adder_2.get_out_socket("output")

        runs = [[target_socket], [target_socket_2]]

        run_graphs = self.graph_converter.to_executable(runs)

        self.assertEqual(2, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())

        self.assertEqual([13.32], run_graphs[1].execute())

    def testExecutesMultipleTargetsInRun(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        component_spec.name = "adder_2"
        adder_2 = self.component_repository.get(component_spec)[0]
        target_socket_2 = adder_2.get_out_socket("output")

        runs = [[target_socket, target_socket_2]]

        run_graphs = self.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.15, 13.32], run_graphs[0].execute())

    def testTensorflowPartsContracted(self):
        filename = "tensorflow_unit_test_blocks/add_constants_tensorflow.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual(2, run_graphs[0].count_components())
        self.assertEqual([8.15], run_graphs[0].execute())

    def testIrrelevantTensorflowPartsExcluded(self):
        pass

    def testIrrelevantTensorflowPartsExcludedWhileRelevantInAdjacentRuns(self):
        pass