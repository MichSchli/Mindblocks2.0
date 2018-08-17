import unittest

from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestTensorflowSimpleBlocks(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testLadderAdd(self):
        filename = "tensorflow_unit_test_blocks/ladder_add_tensorflow.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder_3"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)
        self.setup_holder.initialization_helper.initialize(run_graphs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.0], run_graphs[0].execute())

    def testSecondLadderAdd(self):
        filename = "tensorflow_unit_test_blocks/ladder_add_tensorflow_2.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder_3"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)
        self.setup_holder.initialization_helper.initialize(run_graphs)

        self.assertEqual(1, len(run_graphs))
        self.assertAlmostEqual(8.0, run_graphs[0].execute()[0], places=5)