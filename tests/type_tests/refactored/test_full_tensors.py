import unittest

from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class TestTypeUsage(unittest.TestCase):

    def testCreateValuePreservesDimensions(self):
        tensor_type = SoftTensorTypeModel([5,7])
        tensor_value = tensor_type.initialize_value_model(language="python")

        dimensions = tensor_value.get_dimensions()
        lengths_by_dimension = tensor_value.get_lengths()
        max_lengths_by_dimension = tensor_value.get_max_lengths()

        self.assertListEqual(dimensions, [5,7])
        self.assertListEqual(lengths_by_dimension, [None,None])
        self.assertListEqual(max_lengths_by_dimension, [None,None])

    def testCreateValuePreservesDimensionsWithPlaceholder(self):
        tensor_type = SoftTensorTypeModel([5,None,3])
        tensor_value = tensor_type.initialize_value_model(language="python")

        dimensions = tensor_value.get_dimensions()
        lengths_by_dimension = tensor_value.get_lengths()
        max_lengths_by_dimension = tensor_value.get_max_lengths()

        self.assertListEqual(dimensions, [5,None,3])
        self.assertListEqual(lengths_by_dimension, [None,None,None])
        self.assertListEqual(max_lengths_by_dimension, [None,None, None])

    def testCreateValuePreservesDimensionsScalar(self):
        tensor_type = SoftTensorTypeModel([])
        tensor_value = tensor_type.initialize_value_model(language="python")

        dimensions = tensor_value.get_dimensions()
        lengths_by_dimension = tensor_value.get_lengths()
        max_lengths_by_dimension = tensor_value.get_max_lengths()

        self.assertListEqual(dimensions, [])
        self.assertListEqual(lengths_by_dimension, [])
        self.assertListEqual(max_lengths_by_dimension, [])