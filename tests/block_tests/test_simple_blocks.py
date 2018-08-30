import unittest

from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestSimpleBlocks(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testAdder(self):
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

    def testAdderWithVariable(self):
        filename = "add_constants_with_variable.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket], [target_socket]]
        run_modes = ["train", "test"]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs, run_modes=run_modes)

        self.assertEqual(2, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())
        self.assertEqual([13.15], run_graphs[1].execute())

    def testAdderWithEscapeChars(self):
        filename = "add_constants_with_escape_chars.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket], [target_socket]]
        run_modes = ["train", "test"]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs, run_modes=run_modes)

        self.assertEqual(2, len(run_graphs))
        self.assertEqual([8.15], run_graphs[0].execute())
        self.assertEqual([13.15], run_graphs[1].execute())

        vs = self.setup_holder.variable_repository.get_by_name("constant_2_<value")
        self.assertEqual(1, len(vs))

    def testLadderAdd(self):
        filename = "ladder_add.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder_3"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([8.0], run_graphs[0].execute())

    def testMultiplyByAdding(self):
        filename = "multiply_by_adding.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "adder_3"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([12.4], run_graphs[0].execute())