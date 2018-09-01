import unittest

from Mindblocks.interface import BasicInterface
from tests.setup_holder import SetupHolder


class TestToyRelationPrediction(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

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

        print(interface.predict())