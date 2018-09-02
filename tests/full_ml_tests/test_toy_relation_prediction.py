import unittest

from Mindblocks.interface import BasicInterface
from tests.setup_holder import SetupHolder


class TestToyRelationPrediction(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testPredictEmbeddings(self):
        interface = BasicInterface()

        block_filename = "full_ml_tests/relation_prediction/predict_embeddings.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(block_filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("relation_prediction/toy-125/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/glove.6B.100d.txt")
        interface.load_file(block_filepath)

        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_filepath", embedding_filepath)
        interface.initialize()

        embs = interface.predict()

        self.assertEqual(51, len(embs))

        self.assertEqual(81, len(embs[0]))
        self.assertEqual(60, len(embs[1]))

        self.assertEqual(181, len(embs[0][0]))

    def testValidateWithoutTraining(self):
        interface = BasicInterface()

        block_filename = "full_ml_tests/relation_prediction/pure_relation_prediction.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(block_filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("relation_prediction/toy-125/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/glove.6B.100d.txt")
        interface.load_file(block_filepath)

        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_filepath", embedding_filepath)
        interface.initialize()

        val = interface.validate()
        self.assertLess(0.5, val)

    def testPredictWithoutTraining(self):
        interface = BasicInterface()

        block_filename = "full_ml_tests/relation_prediction/pure_relation_prediction.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(block_filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("relation_prediction/toy-125/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/glove.6B.100d.txt")
        interface.load_file(block_filepath)

        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_filepath", embedding_filepath)
        interface.initialize()

        pred = interface.predict()
        self.assertEqual(51, len(pred))
        self.assertIn("->", pred[0])

    def testTrainAndValidate(self):
        interface = BasicInterface()

        block_filename = "full_ml_tests/relation_prediction/pure_relation_prediction.xml"

        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(block_filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_data_path("relation_prediction/toy-125/")
        embedding_filepath = self.setup_holder.filepath_handler.get_test_data_path("embeddings/glove.6B.100d.txt")
        interface.load_file(block_filepath)

        interface.set_variable("data_folder", data_filepath)
        interface.set_variable("embedding_filepath", embedding_filepath)
        interface.initialize()

        interface.train()

        val = interface.validate()
        self.assertGreater(0.01, val)