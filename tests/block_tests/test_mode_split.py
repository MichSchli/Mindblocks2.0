import unittest

from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestModeSplitter(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testModeSplitCorrectValues(self):
        filename = "mode_pull_constants.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "mode_split"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket], [target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs, run_modes=["train", "test"])

        self.assertEqual(2, len(run_graphs))
        self.assertEqual([5.17], run_graphs[0].execute())
        self.assertEqual([2.98], run_graphs[1].execute())

    def testModeSplitOnlyUsedPartsPulled(self):
        filename = "mode_pull_constants.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "mode_split"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket], [target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs, run_modes=["train", "test"])

        self.assertEqual(2, len(run_graphs[0].get_components()))
        self.assertEqual(2, len(run_graphs[1].get_components()))
