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
import numpy as np


class TestIrisBlocks(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testReadsData(self):
        filename = "iris_tests/read_iris_data.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "data_splitter"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket_1 = adder.get_out_socket("left")
        target_socket_2 = adder.get_out_socket("right")

        runs = [[target_socket_1, target_socket_2]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)
        result = run_graphs[0].execute()

        self.assertEqual("0.2", result[0][0][-1])
        self.assertEqual("Iris-setosa", result[1][0][0])

    def testIndexesNames(self):
        filename = "iris_tests/read_and_index_iris_data.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "indexer"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket_1 = adder.get_out_socket("output")

        runs = [[target_socket_1]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)
        result = run_graphs[0].execute()

        self.assertEqual('0', result[0][0][0])
        self.assertEqual('1', result[0][50][0])
        self.assertEqual('2', result[0][100][0])

    def testBatchesForTraining(self):
        filename = "iris_tests/batch_iris.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "data_batcher"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket_1 = adder.get_out_socket("output")

        runs = [[target_socket_1]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)
        run_graphs[0].init_batches()

        result = run_graphs[0].execute()[0]
        self.assertTrue(run_graphs[0].has_batches())
        self.assertEqual(50, len(result))
        result = run_graphs[0].execute()[0]
        self.assertTrue(run_graphs[0].has_batches())
        self.assertEqual(50, len(result))
        result = run_graphs[0].execute()[0]
        self.assertEqual(50, len(result))
        self.assertFalse(run_graphs[0].has_batches())

        run_graphs[0].init_batches()

        result = run_graphs[0].execute()[0]
        self.assertTrue(run_graphs[0].has_batches())
        self.assertEqual(50, len(result))
        result = run_graphs[0].execute()[0]
        self.assertTrue(run_graphs[0].has_batches())
        self.assertEqual(50, len(result))
        result = run_graphs[0].execute()[0]
        self.assertEqual(50, len(result))
        self.assertFalse(run_graphs[0].has_batches())

    def testAccuracyOnlyWithoutMlHelper(self):
        filename = "iris_tests/untrained_iris_accuracy.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "accuracy"
        component = self.setup_holder.component_repository.get(component_spec)[0]
        accuracy = component.get_out_socket("output")

        runs = [[accuracy]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)
        run_graphs[0].init_batches()
        performance = run_graphs[0].execute()[0]

        self.assertGreaterEqual(10.0, performance)
        self.assertLessEqual(0.0, performance)

    def testAccuracyOnly(self):
        filename = "iris_tests/untrained_iris_accuracy.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "accuracy"
        component = self.setup_holder.component_repository.get(component_spec)[0]
        accuracy = component.get_out_socket("output")

        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper(evaluate=accuracy)

        performance = ml_helper.evaluate()

        self.assertGreaterEqual(1.0, performance)
        self.assertLessEqual(0.0, performance)
        self.assertFalse(ml_helper.evaluate_function.has_batches())

    def testFullTraining(self):
        filename = "iris_tests/full_iris.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adam_upd"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        update = adder.get_out_socket("update")

        component_spec = CreationComponentSpecifications()
        component_spec.name = "cross_ent"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        loss = adder.get_out_socket("output")

        component_spec = CreationComponentSpecifications()
        component_spec.name = "accuracy"
        component = self.setup_holder.component_repository.get(component_spec)[0]
        accuracy = component.get_out_socket("output")

        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper(update=update, loss=loss, evaluate=accuracy)

        ml_helper.train()
        performance = ml_helper.evaluate()

        print(performance)

        self.assertGreaterEqual(1.0, performance)
        self.assertLess(0.9, performance)

    def testFullTrainingWithStoredPointers(self):
        filename = "iris_tests/full_iris.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()
        performance = ml_helper.evaluate()

        self.assertLess(0.9, performance)

    def testFullTrainingWithDropouts(self):
        filename = "iris_tests/full_iris_with_dropouts.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()
        performance = ml_helper.evaluate()

        self.assertLess(0.9, performance)

    def testFullTrainingWithValidation(self):
        filename = "iris_tests/full_iris_with_separate_validation.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()
        performance = ml_helper.evaluate()

        self.assertLess(0.9, performance)

    def testFullTrainingWithThreeDataSets(self):
        filename = "iris_tests/full_iris_with_three_datasets.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()
        performance = ml_helper.evaluate()

        self.assertLess(0.9, performance)