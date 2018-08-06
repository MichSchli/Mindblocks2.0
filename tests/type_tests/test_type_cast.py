import unittest

from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class TestTypeUsage(unittest.TestCase):

    def testCastScalarFloatTensorType(self):
        type = TensorTypeModel("string", [])
        value = type.initialize_value_model()
        value.assign("5.67")

        value = value.cast("float")

        self.assertEqual(5.67, value.get_value())

    def testCastScalarIntTensorType(self):
        type = TensorTypeModel("string", [])
        value = type.initialize_value_model()
        value.assign("5")

        value = value.cast("int")

        self.assertEqual(5, value.get_value())