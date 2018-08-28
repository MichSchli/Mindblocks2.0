import os
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

    def testAlwaysUsesMarkedGraph(self):
        interface = BasicInterface()

        filename = "iris_tests/full_iris_with_extra_graphs.xml"
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

    def testProfileTrain(self):
        interface = BasicInterface()

        filename = "iris_tests/full_iris.xml"
        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "iris_tests")

        profile_dir = self.setup_holder.filepath_handler.get_test_data_path("test_output/logs")

        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.initialize(profile=True, log_dir=profile_dir)

        interface.train()
        performance = interface.evaluate()

        self.assertGreaterEqual(1.0, performance)
        self.assertLess(0.9, performance)

        self.assertTrue(os.path.exists(profile_dir + "/timeline.json"))