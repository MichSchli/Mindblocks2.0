import unittest

from Mindblocks.model.variable.variable_model import VariableModel

class TestVariableModel(unittest.TestCase):

    def testGetValueSpecificMode(self):
        element_1 = VariableModel()
        element_1.name = "testElement"
        element_1.set_value("45", mode="validate")
        element_1.set_value("547", mode="test")

        self.assertEqual(None, element_1.get_value(mode=None))
        self.assertEqual(None, element_1.get_value(mode="train"))
        self.assertEqual("45", element_1.get_value(mode="validate"))
        self.assertEqual("547", element_1.get_value(mode="test"))

    def testGetValueSpecificModeWithDefault(self):
        element_1 = VariableModel()
        element_1.name = "testElement"
        element_1.set_value("233", mode=None)
        element_1.set_value("45", mode="validate")
        element_1.set_value("547", mode="test")

        self.assertEqual("233", element_1.get_value(mode=None))
        self.assertEqual("233", element_1.get_value(mode="train"))
        self.assertEqual("45", element_1.get_value(mode="validate"))
        self.assertEqual("547", element_1.get_value(mode="test"))

    def testReplaceInString(self):
        element_1 = VariableModel()
        element_1.name = "testElement"
        element_1.set_value("test value", mode="validate")

        source_str = "will this $testElement have the correct value"
        target_str = element_1.replace_in_string(source_str, mode="validate")

        self.assertEqual("will this test value have the correct value", target_str)

    def testReplaceInStringMultipleModes(self):
        element_1 = VariableModel()
        element_1.name = "testElement"
        element_1.set_value("test value", mode="validate")
        element_1.set_value("other test value", mode="test")

        source_str = "will this $testElement have the correct value"
        target_str = element_1.replace_in_string(source_str, mode="test")

        self.assertEqual("will this other test value have the correct value", target_str)