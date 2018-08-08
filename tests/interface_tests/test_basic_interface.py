import unittest

from Mindblocks.interface import BasicInterface
from tests.setup_holder import SetupHolder


class TestBasicInterface(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testIrisTrainAndEvaluate(self):
        interface = BasicInterface()

        filename = "iris_tests/full_iris.xml"
        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "iris_tests")

        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.initialize()

        interface.train()
        performance = interface.evaluate()

        self.assertGreaterEqual(1.0, performance)
        self.assertLess(0.9, performance)