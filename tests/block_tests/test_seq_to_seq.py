import unittest

from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestSeqToSeqBlocks(unittest.TestCase):

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
        output = run_graphs[0].execute()[0]
        self.assertEqual(5.17, output[0])
        self.assertEqual(2.98, output[1])

    def testRepeatRnn(self):
        filename = "seq_to_seq_tests/repeat_rnn.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)


        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "cell"
        target_c = self.setup_holder.component_repository.get(component_spec)[0]

        target_socket = target_c.get_out_socket("output_sequences")

        runs = [[target_socket]]
        run_graphs = self.setup_holder.graph_converter.to_executable(runs)

        self.assertEqual(1, len(run_graphs))
        output = run_graphs[0].execute()[0]

        should_be = [[[0,0], [0,1], [1,2], [2,3], [3,4]],
                     [[0,5], [5,1], [1,0], [0,4]]]

        for i in range(len(should_be)):
            for j in range(len(should_be[i])):
                for k in range(len(should_be[i][j])):
                    self.assertEqual(should_be[i][j][k], output[i][j][k])

    def testBasicLanguageModel(self):
        filename = "seq_to_seq_tests/basic_language_model.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        loss = ml_helper.validate()

        self.assertLess(0.0, loss)
        self.assertGreater(0.05, loss)

    def testBasicLanguageModelWithEmbedding(self):
        filename = "seq_to_seq_tests/basic_language_model_with_embedding.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)
        self.setup_holder.variable_repository.set_variable_value("data_file", "test_conll_file.conll")

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        loss = ml_helper.validate()

        self.assertLess(0.0, loss)
        self.assertGreater(0.05, loss)

    def testBasicLanguageModelWithEmbeddingTenSentences(self):
        filename = "seq_to_seq_tests/basic_language_model_with_embedding.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)
        self.setup_holder.variable_repository.set_variable_value("data_file", "larger_test_conll_file.conll")

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        loss = ml_helper.validate()

        self.assertLess(0.0, loss)
        self.assertGreater(0.35, loss)

    def testBasicLanguageModelLstm(self):
        filename = "seq_to_seq_tests/basic_language_model_lstm.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)
        self.setup_holder.variable_repository.set_variable_value("data_file", "larger_test_conll_file.conll")

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        loss = ml_helper.validate()

        self.assertLess(0.0, loss)
        self.assertGreater(0.05, loss)

    def testBasicLanguageModelStopAndStart(self):
        filename = "seq_to_seq_tests/basic_language_model_stop_and_start_tokens.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)
        self.setup_holder.variable_repository.set_variable_value("data_file", "larger_test_conll_file.conll")

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        loss = ml_helper.validate()

        self.assertLess(0.0, loss)
        self.assertGreater(0.05, loss)

    def testBeamSearchAllBeams(self):
        filename = "seq_to_seq_tests/basic_language_model_with_beam_search_top_3.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)
        self.setup_holder.variable_repository.set_variable_value("data_file", "larger_test_conll_file.conll")

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        predictions = ml_helper.predict()
        gold_sentences = ["this is a sentence . _STOP_",
                          "so is this . _STOP_",
                          "also this . _STOP_",
                          "this is also a sentence . _STOP_",
                          "also this is a sentence . _STOP_",
                          "sentence . . . _STOP_",
                          "this . is . a . sentence . _STOP_",
                          "this . _STOP_",
                          "this sentence . _STOP_",
                          "this is a sentence also . _STOP_"]

        self.assertEqual(30, len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i*3])
            self.assertEqual(s, pred_sent)

            pred_sent = " ".join(predictions[i*3 + 1])
            self.assertNotEqual(s, pred_sent)

            pred_sent = " ".join(predictions[i*3 + 2])
            self.assertNotEqual(s, pred_sent)

    def testBeamSearchBestBeam(self):
        filename = "seq_to_seq_tests/basic_language_model_with_beam_search_top_1.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)
        self.setup_holder.variable_repository.set_variable_value("data_file", "larger_test_conll_file.conll")

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        predictions = ml_helper.predict()
        gold_sentences = ["this is a sentence . _STOP_",
                          "so is this . _STOP_",
                          "also this . _STOP_",
                          "this is also a sentence . _STOP_",
                          "also this is a sentence . _STOP_",
                          "sentence . . . _STOP_",
                          "this . is . a . sentence . _STOP_",
                          "this . _STOP_",
                          "this sentence . _STOP_",
                          "this is a sentence also . _STOP_"]

        self.assertEqual(10, len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testLanguageModelWithGloveEmbeddings(self):
        filename = "seq_to_seq_tests/basic_language_model_glove_embeddings.xml"

        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "conll_reader_tests")
        self.setup_holder.variable_repository.set_variable_value("data_folder", data_filepath)
        self.setup_holder.variable_repository.set_variable_value("data_file", "larger_test_conll_file.conll")

        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings")
        self.setup_holder.variable_repository.set_variable_value("embedding_folder", embedding_filepath)

        component_spec = CreationComponentSpecifications()
        c = self.setup_holder.component_repository.get(component_spec)[0]
        graph = c.get_graph()
        ml_helper = self.setup_holder.ml_helper_factory.build_ml_helper_from_graph(graph)

        ml_helper.train()

        predictions = ml_helper.predict()
        gold_sentences = ["this is a sentence . EOS",
                          "so is this . EOS",
                          "also this . EOS",
                          "this is also a sentence . EOS",
                          "also this is a sentence . EOS",
                          "sentence . . . EOS",
                          "this . is . a . sentence . EOS",
                          "this . EOS",
                          "this sentence . EOS",
                          "this is a sentence also . EOS"]

        self.assertEqual(10, len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)