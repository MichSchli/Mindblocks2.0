import unittest

from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestSimpleBlocks(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testCellSanity(self):
        filename = "seq_to_seq_tests/repeat_rnn_cell_sanity.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "cell"
        target_c = self.setup_holder.component_repository.get(component_spec)[0]

        target_socket = target_c.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([[5.17, 2.98]], run_graphs[0].execute())