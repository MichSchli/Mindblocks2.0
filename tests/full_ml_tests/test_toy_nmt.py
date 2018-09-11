import os
import shutil
import unittest

from Mindblocks.interface import BasicInterface
from Mindblocks.repository.creation_component_repository.creation_component_specifications import \
    CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestToyNmt(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testBasicSeqtoSeq(self):
        filename = "full_ml_tests/toy_nmt/basic_toy_nmt.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testBasicSeqtoSeqPerplexity(self):
        filename = "full_ml_tests/toy_nmt/basic_toy_nmt_with_perplexity.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        initial_validation = interface.validate()
        interface.train()
        second_validation = interface.validate()

        self.assertLess(50, initial_validation)
        self.assertLess(1, second_validation)
        self.assertGreater(1.01, second_validation)

    def testSeqtoSeqWithKVAttention(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_attention.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithLuongKVAttention(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_luong_kv_attention.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithLuongMulAttention(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_luong_mul_attention.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithLuongMulAttentionParameterCount(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_luong_mul_attention.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        ml_helper = interface.ml_helper
        params = ml_helper.count_parameters()

        self.assertEqual(376254, params)

    def testSeqtoSeqWithBatches(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_batches.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithOrderedBatches(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_ordered_batches.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithBatchesNoAveragingLoss(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_batches_averaging.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithSgdLrDecease(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_sgd_learning_rate_decay.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        upd_component = interface.get_execution_component("upd")
        learning_rate_variable = upd_component.value_model.get_learning_rate()
        lr = interface.ml_helper.tensorflow_session_model.run(learning_rate_variable, {})
        self.assertAlmostEqual(0.1, lr)

        interface.train(iterations=10)
        lr = interface.ml_helper.tensorflow_session_model.run(learning_rate_variable, {})
        self.assertAlmostEqual(0.1, lr)

        interface.train(iterations=1991)
        lr = interface.ml_helper.tensorflow_session_model.run(learning_rate_variable, {})
        self.assertAlmostEqual(0.0999, lr)

        interface.train(iterations=1)
        lr = interface.ml_helper.tensorflow_session_model.run(learning_rate_variable, {})
        self.assertAlmostEqual(0.0998001, lr)

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithUnks(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_unks.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)

        interface.initialize()

        f = open(data_filepath + "tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        unk_count = 0

        for i, s in enumerate(gold_sentences):
            gold_tokens = s.split(" ")
            for j, t in enumerate(gold_tokens):
                if predictions[i][j] != "UNK":
                    self.assertEqual(t, predictions[i][j])
                else:
                    unk_count += 1

        self.assertGreater(10, unk_count)

    def testSeqtoSeqWithUnkReplacement(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_replace_unks.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.add_console_logger({"formatting" : ["token_replacement"]})
        interface.initialize()

        f = open(data_filepath + "tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        unk_count = 0
        missed_count = 0

        for i, s in enumerate(gold_sentences):
            gold_tokens = s.split(" ")
            for j, t in enumerate(gold_tokens):
                if predictions[i][j] != "UNK":
                    if predictions[i][j] != t:
                        missed_count += 1
                else:
                    unk_count += 1

        self.assertEqual(0, unk_count)
        self.assertGreater(10, missed_count)

    def testSeqtoSeqWithKVMultiHeadAttention(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_attention.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.set_variable("attention_heads", "5")
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithEncoderFinalState(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_init_decoder_from_encoder.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithMultipleEncoderLayers(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_multiple_lstm_layers.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithMultipleDecoderLayers(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_multiple_output_lstm_layers.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithVariableDecoderLayers(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_variable_output_lstm_layers.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.set_variable("lstm_layers", "3")
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.set_variable("lstm_layers", "1")
        interface.initialize()

        f = open(data_filepath + "tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithMultipleDecoderLayersInitFromDec(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_multiple_output_lstm_layers_feed_enc.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testSeqtoSeqWithLuongMulAttentionSaveAndLoad(self):
        filename = "full_ml_tests/toy_nmt/toy_nmt_luong_mul_attention.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath+"tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

        model_filepath = self.setup_holder.filepath_handler.get_test_data_path("stored_models/toy_nmt/luong")
        if os.path.exists(model_filepath):
            shutil.rmtree(model_filepath)

        interface.save(model_filepath)

        interface_2 = BasicInterface()
        interface_2.load_file(block_filepath)
        interface_2.set_variable("data_folder", data_filepath)
        interface_2.set_variable("embedding_folder", embedding_filepath)
        interface_2.initialize()
        interface_2.load(model_filepath)

        predictions = interface_2.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testVariationalLuongMulAttention(self):
        filename = "full_ml_tests/toy_nmt/variational_toy_nmt_luong_mul_attention.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath + "tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testVariationalLuongMulAttentionMean(self):
        filename = "full_ml_tests/toy_nmt/variational_toy_nmt_luong_mul_attention_mean.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath + "tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)

    def testVariationalLuongMulAttentionWithConditionalPrior(self):
        filename = "full_ml_tests/toy_nmt/variational_toy_nmt_conditional_prior.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_folder", embedding_filepath)
        interface.initialize()

        f = open(data_filepath + "tgt.txt")
        lines = [l.strip() for l in f]
        gold_sentences = [l + " EOS" for l in lines]
        f.close()

        interface.train()
        predictions = interface.predict()

        self.assertEqual(len(gold_sentences), len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i])
            self.assertEqual(s, pred_sent)