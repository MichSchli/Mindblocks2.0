import os
import shutil
import unittest

from Mindblocks.interface import BasicInterface
from Mindblocks.repository.creation_component_repository.creation_component_specifications import \
    CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestPointerNetwork(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testReverseSequenceTaskNoPointer(self):
        filename = "full_ml_tests/pointer_network/no_pointer.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/reverse_sequence/")
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

    def testReverseSequenceTaskWithOnlyPointer(self):
        filename = "full_ml_tests/pointer_network/only_pointer.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/reverse_sequence/")
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