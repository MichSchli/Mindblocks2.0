import unittest

from Mindblocks.repository.creation_component_repository.creation_component_specifications import \
    CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestConllReaderBlocks(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testConllReaderReadsFile(self):
        filename = "conll_reader_tests/conll_reader.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        conll_filepath = self.setup_holder.filepath_handler.get_test_block_path("conll_reader_tests/test_conll_file.conll")
        self.setup_holder.variable_repository.set_variable_value("data_file", conll_filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "reader"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        self.assertEqual([[[[0, "this"],
                           [1, "is"],
                           [2, "a"],
                           [3, "sentence"],
                           [4, "."]],
                          [[0, "so"],
                           [1, "is"],
                           [2, "this"],
                           [3, "."]]]], run_graphs[0].execute())

    def testDeIndexer(self):
        filename = "conll_reader_tests/conll_reader_with_indexer.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)
        conll_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests/test_conll_file.conll")
        self.setup_holder.variable_repository.set_variable_value("data_file", conll_filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "deindexer"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = adder.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        results = run_graphs[0].execute()
        self.assertEqual([[["this",
                           "is",
                           "a",
                           "sentence",
                           "."],
                          ["so",
                           "is",
                           "this",
                           "."]]], results)

    def testEmbeddingIndexing(self):
        filename = "conll_reader_tests/embedding_index.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        conll_filepath = self.setup_holder.filepath_handler.get_test_block_path("conll_reader_tests/test_conll_file.conll")
        self.setup_holder.variable_repository.set_variable_value("data_file", conll_filepath)

        embedding_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests/test_embeddings.txt")
        self.setup_holder.variable_repository.set_variable_value("embedding_file", embedding_filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "indexer"
        target = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = target.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))

        self.assertEqual([[[0,1,2,3,4],
                          [5,1,0,4]]], run_graphs[0].execute())
        
    def testEmbeddingLookup(self):
        filename = "conll_reader_tests/embedding_lookup.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        conll_filepath = self.setup_holder.filepath_handler.get_test_block_path("conll_reader_tests/test_conll_file.conll")
        self.setup_holder.variable_repository.set_variable_value("data_file", conll_filepath)

        embedding_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests/test_embeddings.txt")
        self.setup_holder.variable_repository.set_variable_value("embedding_file", embedding_filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "average"
        target = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket = target.get_out_socket("output")

        runs = [[target_socket]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        result = run_graphs[0].execute()
        self.assertEqual(2, len(result[0]))

        print(result)