import os
import shutil
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

    def testLoggerSavesLog(self):
        interface = BasicInterface()

        filename = "iris_tests/full_iris.xml"
        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "iris_tests")

        logger_filepath = self.setup_holder.filepath_handler.get_test_data_path("test_output/logs/iris/log.txt")
        log_dir_filepath = self.setup_holder.filepath_handler.get_test_data_path("test_output/logs/iris/")
        if os.path.exists(log_dir_filepath):
            shutil.rmtree(log_dir_filepath)

        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.initialize()

        logger_config = {"training": ["status",
                                       "loss",
                                       "parameters"],
                          "validation": ["all"]}
        interface.add_file_logger(logger_config, logger_filepath)

        interface.train()
        performance = interface.evaluate()

        self.assertGreaterEqual(1.0, performance)
        self.assertLess(0.9, performance)

        self.assertTrue(os.path.exists(logger_filepath))

    def testIrisSaveAndLoad(self):
        interface = BasicInterface()

        filename = "iris_tests/full_iris_no_shuffling.xml"
        block_filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        data_filepath = self.setup_holder.filepath_handler.get_test_block_path(
            "iris_tests")

        model_filepath = self.setup_holder.filepath_handler.get_test_data_path("stored_models/iris/iris.model")
        if os.path.exists(model_filepath):
            shutil.rmtree(model_filepath)

        interface.load_file(block_filepath)
        interface.set_variable("data_folder", data_filepath)
        interface.initialize()

        interface.train()
        performance = interface.evaluate()

        self.assertGreaterEqual(1.0, performance)
        self.assertLess(0.9, performance)

        interface.save(model_filepath)

        interface_2 = BasicInterface()
        interface_2.load_file(block_filepath)
        interface_2.set_variable("data_folder", data_filepath)
        interface_2.initialize()
        interface_2.load(model_filepath)

        performance = interface_2.evaluate()

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