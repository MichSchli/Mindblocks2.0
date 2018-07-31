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
from tests.setup_holder import SetupHolder


class TestGraphConverter(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testInitializingValues(self):
        filename = "add_constants.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        value_dictionary = self.setup_holder.graph_converter.build_value_dictionary(runs, ["train"])

        self.assertIsNotNone(value_dictionary)
        self.assertEqual(9, len(value_dictionary))

        keys = list(value_dictionary.keys())

        for component in list(self.setup_holder.component_repository.elements.values()):
            self.assertIn(str(component.identifier) + "train", keys)
            self.assertIsNotNone(value_dictionary[str(component.identifier) + "train"])

    def testExcludesIrrelevantPartsFromValues(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        value_dictionary = self.setup_holder.graph_converter.build_value_dictionary(runs, ["train"])

        self.assertIsNotNone(value_dictionary)
        self.assertEqual(9, len(value_dictionary))

        keys = list(value_dictionary.keys())

        for component in list(self.setup_holder.component_repository.elements.values()):
            if component.name != "adder_2":
                self.assertIn(str(component.identifier) + "train", keys)
                self.assertIsNotNone(value_dictionary[str(component.identifier) + "train"])

    def testCreatesExecutionGraphs(self):
        filename = "add_constants.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())

    def testExcludesIrrelevantPartsFromExecution(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())

    def testExecutesMultipleRuns(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        component_spec.name = "adder_2"
        adder_2 = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket_2 = adder_2.get_out_socket("output")

        runs = [[target_socket], [target_socket_2]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(2, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())

        self.assertEqual([13.32], run_graphs[1].execute())

    def testExecutesMultipleTargetsInRun(self):
        filename = "add_constants_with_extra_adder.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        component_spec.name = "adder_2"
        adder_2 = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket_2 = adder_2.get_out_socket("output")

        runs = [[target_socket, target_socket_2]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.15, 13.32], run_graphs[0].execute())

    def testTensorflowPartsContracted(self):
        filename = "tensorflow_unit_test_blocks/add_constants_tensorflow.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual(2, run_graphs[0].count_components())
        self.assertAlmostEqual(8.15, run_graphs[0].execute()[0], places=5)

    def testIrrelevantTensorflowPartsExcluded(self):
        pass

    def testIrrelevantTensorflowPartsExcludedWhileRelevantInAdjacentRuns(self):
        pass