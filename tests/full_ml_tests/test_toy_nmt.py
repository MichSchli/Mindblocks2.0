import unittest

from Mindblocks.interface import BasicInterface
from Mindblocks.repository.creation_component_repository.creation_component_specifications import \
    CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestSimpleBlocks(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testBasicSeqtoSeq(self):
        filename = "full_ml_tests/toy_nmt/basic_toy_nmt.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("nmt/toy/")

        interface = BasicInterface()
        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.initialize()

        interface.train()
        predictions = interface.predict()

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

        self.assertEqual(len(gold_sentences) * 3, len(predictions))

        for i, s in enumerate(gold_sentences):
            pred_sent = " ".join(predictions[i*3])
            self.assertEqual(s, pred_sent)

            pred_sent = " ".join(predictions[i*3 + 1])
            self.assertNotEqual(s, pred_sent)

            pred_sent = " ".join(predictions[i*3 + 2])
            self.assertNotEqual(s, pred_sent)