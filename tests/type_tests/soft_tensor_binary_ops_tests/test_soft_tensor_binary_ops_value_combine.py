import unittest

from Mindblocks.helpers.soft_tensors.soft_tensor_binary_operator_helper import SoftTensorBinaryOperatorHelper
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel
import numpy as np


class TestSoftTensorBinaryOpsValueCombine(unittest.TestCase):

    def testProcessScalar(self):
        tensor_type = SoftTensorTypeModel([], string_type="int")
        tensor_value = tensor_type.initialize_value_model(language="python")
        tensor_value.initial_assign(5)

        tensor_type_2 = SoftTensorTypeModel([], string_type="int")
        tensor_value_2 = tensor_type_2.initialize_value_model(language="python")
        tensor_value_2.initial_assign(5)

        helper = SoftTensorBinaryOperatorHelper()

        output_type = helper.create_output_type(tensor_type, tensor_type_2, "int", "test_add")
        output_value = output_type.initialize_value_model(language="python")

        op = lambda x,y: x + y
        result = helper.process(tensor_value, tensor_value_2, op, output_value, language="python")

        self.assertEqual(result, output_value)
        self.assertEqual(10, result.get_value())
        self.assertEqual([], result.get_lengths())

    def testProcessVectors(self):
        tensor_type = SoftTensorTypeModel([2], string_type="int")
        tensor_value = tensor_type.initialize_value_model(language="python")
        tensor_value.initial_assign(np.array([2,3]))

        tensor_type_2 = SoftTensorTypeModel([2], string_type="int")
        tensor_value_2 = tensor_type_2.initialize_value_model(language="python")
        tensor_value_2.initial_assign(np.array([5,-1]))

        helper = SoftTensorBinaryOperatorHelper()

        output_type = helper.create_output_type(tensor_type, tensor_type_2, "int", "test_add")
        output_value = output_type.initialize_value_model(language="python")

        op = lambda x,y: x + y
        result = helper.process(tensor_value, tensor_value_2, op, output_value, language="python")

        self.assertEqual(result, output_value)
        self.numpyAssertEqual(np.array([7,2], dtype=np.int32), result.get_value())

    def testProcessExpand(self):
        tensor_type = SoftTensorTypeModel([1], string_type="int")
        tensor_value = tensor_type.initialize_value_model(language="python")
        tensor_value.initial_assign(np.array([2]))

        tensor_type_2 = SoftTensorTypeModel([2], string_type="int")
        tensor_value_2 = tensor_type_2.initialize_value_model(language="python")
        tensor_value_2.initial_assign(np.array([5,-1]))

        helper = SoftTensorBinaryOperatorHelper()

        output_type = helper.create_output_type(tensor_type, tensor_type_2, "int", "test_add")
        output_value = output_type.initialize_value_model(language="python")

        op = lambda x,y: x + y
        result = helper.process(tensor_value, tensor_value_2, op, output_value, language="python")

        self.assertEqual(result, output_value)
        self.numpyAssertEqual(np.array([7,1], dtype=np.int32), result.get_value())

    def testProcessExpandSoft(self):
        tensor_type = SoftTensorTypeModel([2,1], string_type="int")
        tensor_value = tensor_type.initialize_value_model(language="python")
        tensor_value.initial_assign(np.array([[2], [5]]))

        tensor_type_2 = SoftTensorTypeModel([2, 2], string_type="int", soft_by_dimensions=[False, True])
        tensor_value_2 = tensor_type_2.initialize_value_model(language="python")
        tensor_value_2.initial_assign([[1,2], [3]])

        helper = SoftTensorBinaryOperatorHelper()

        output_type = helper.create_output_type(tensor_type, tensor_type_2, "int", "test_add")
        output_value = output_type.initialize_value_model(language="python")

        op = lambda x,y: x + y
        result = helper.process(tensor_value, tensor_value_2, op, output_value, language="python")

        self.assertEqual(result, output_value)
        self.numpyAssertEqual(np.array([[3,4],[8,result.get_value()[1][1]]], dtype=np.int32), result.get_value())
        self.assertEqual(len(result.get_lengths()), 2)
        self.assertIsNone(result.get_lengths()[0])
        self.numpyAssertEqual(np.array([2,1], dtype=np.int32), result.get_lengths()[1])

    def testProcessExpandScalar(self):
        tensor_type = SoftTensorTypeModel([], string_type="int")
        tensor_value = tensor_type.initialize_value_model(language="python")
        tensor_value.initial_assign(2)

        tensor_type_2 = SoftTensorTypeModel([2], string_type="int")
        tensor_value_2 = tensor_type_2.initialize_value_model(language="python")
        tensor_value_2.initial_assign(np.array([5,-1]))

        helper = SoftTensorBinaryOperatorHelper()

        output_type = helper.create_output_type(tensor_type, tensor_type_2, "int", "test_add")
        output_value = output_type.initialize_value_model(language="python")

        op = lambda x,y: x + y
        result = helper.process(tensor_value, tensor_value_2, op, output_value, language="python")

        self.assertEqual(result, output_value)
        self.numpyAssertEqual(np.array([7,1], dtype=np.int32), result.get_value())

    def numpyAssertEqual(self, reference, test):
        reference = reference.flatten()
        test = test.flatten()

        self.assertEqual(reference.dtype, test.dtype)

        for i in range(reference.shape[0]):
            self.assertEqual(reference[i], test[i])