import unittest

from Mindblocks.helpers.soft_tensors.soft_tensor_binary_operator_helper import SoftTensorBinaryOperatorHelper
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class TestSoftTensorBinaryOpsTypeCombine(unittest.TestCase):

    def testCombineHard(self):
        left = SoftTensorTypeModel([5,10,3])
        right = SoftTensorTypeModel([5,10,3])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)

    def testCombineScalarsProducesScalar(self):
        left = SoftTensorTypeModel([])
        right = SoftTensorTypeModel([])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 0)

    def testCombineSingularsDoesNotProduceScalar(self):
        left = SoftTensorTypeModel([1])
        right = SoftTensorTypeModel([1])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 1)
        self.assertEqual([1], combined_dims)

    def testCombineHardExpandLeft(self):
        left = SoftTensorTypeModel([5,1,3,1])
        right = SoftTensorTypeModel([5,10,3,7])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 4)
        self.assertEqual([5,10,3,7], combined_dims)

    def testCombineHardExpandRight(self):
        left = SoftTensorTypeModel([5,10,3,7])
        right = SoftTensorTypeModel([1,10,1,7])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 4)
        self.assertEqual([5,10,3,7], combined_dims)

    def testCombineHardExpandBoth(self):
        left = SoftTensorTypeModel([5,1,3,1])
        right = SoftTensorTypeModel([1,10,1,7])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 4)
        self.assertEqual([5,10,3,7], combined_dims)

    def testCombineHardUnknownDim(self):
        left = SoftTensorTypeModel([5,None,3])
        right = SoftTensorTypeModel([5,10,3])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)

    def testCombineHardMultipleUnknownDims(self):
        left = SoftTensorTypeModel([5,None,3])
        right = SoftTensorTypeModel([None,10,None])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)

    def testCombineHardUnknownDimExpands(self):
        left = SoftTensorTypeModel([5,None,3])
        right = SoftTensorTypeModel([5,1,3])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,None,3], combined_dims)

    def testCombineHardMultipleUnknownDimsExpand(self):
        left = SoftTensorTypeModel([5,None,1])
        right = SoftTensorTypeModel([None,1,None])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,None,None], combined_dims)

    def testCombineHardMutualUnknownStillUnknown(self):
        left = SoftTensorTypeModel([5,None,1])
        right = SoftTensorTypeModel([None,None,None])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,None,None], combined_dims)

    def testCombineDimMismatchGivesException(self):
        left = SoftTensorTypeModel([5,10,3])
        right = SoftTensorTypeModel([5,12,3])

        helper = SoftTensorBinaryOperatorHelper()

        with self.assertRaises(Exception) as context:
            helper.get_combine_type_dimensions(left, right, "test_combine")

    def testCombineSoftnessMismatchGivesException(self):
        left = SoftTensorTypeModel([5,10,3])
        right = SoftTensorTypeModel([5,10,3], soft_by_dimensions=[False, True, False])

        helper = SoftTensorBinaryOperatorHelper()

        with self.assertRaises(Exception) as context:
            helper.get_combine_type_dimensions(left, right, "test_combine")

    def testCombineSoft(self):
        left = SoftTensorTypeModel([5,10,3], soft_by_dimensions=[True, False, True])
        right = SoftTensorTypeModel([5,10,3], soft_by_dimensions=[True, False, True])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, combined_softness = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)
        self.assertEqual(len(combined_softness), 3)
        self.assertEqual([True, False, True], combined_softness)

    def testCombineSoftExpandsToHardSingular(self):
        left = SoftTensorTypeModel([5,10,1], soft_by_dimensions=[True, False, False])
        right = SoftTensorTypeModel([5,10,3], soft_by_dimensions=[True, False, True])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, combined_softness = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)
        self.assertEqual(len(combined_softness), 3)
        self.assertEqual([True, False, True], combined_softness)

    def testCombineSoftExpandsToMultipleHardSingulars(self):
        left = SoftTensorTypeModel([1,10,1], soft_by_dimensions=[False, False, False])
        right = SoftTensorTypeModel([5,10,3], soft_by_dimensions=[True, False, True])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, combined_softness = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)
        self.assertEqual(len(combined_softness), 3)
        self.assertEqual([True, False, True], combined_softness)

    def testCombineSoftnessCannotExpandToSingularSoft(self):
        left = SoftTensorTypeModel([5,10,1], soft_by_dimensions=[True, False, True])
        right = SoftTensorTypeModel([5,10,3], soft_by_dimensions=[True, False, True])

        helper = SoftTensorBinaryOperatorHelper()

        with self.assertRaises(Exception) as context:
            helper.get_combine_type_dimensions(left, right, "test_combine")

    def testCombineSoftUnknownDims(self):
        left = SoftTensorTypeModel([None,10,3], soft_by_dimensions=[True, False, True])
        right = SoftTensorTypeModel([5,10,None], soft_by_dimensions=[True, False, True])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, combined_softness = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)
        self.assertEqual(len(combined_softness), 3)
        self.assertEqual([True, False, True], combined_softness)

    def testCombineHardPartial(self):
        left = SoftTensorTypeModel([5,10])
        right = SoftTensorTypeModel([5,10,3])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)

    def testCombineHardLongPartial(self):
        left = SoftTensorTypeModel([5,10])
        right = SoftTensorTypeModel([5,10,3,10,2,5])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, _ = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 6)
        self.assertEqual([5,10,3,10,2,5], combined_dims)

    def testCombineSoftPartial(self):
        left = SoftTensorTypeModel([5,10], soft_by_dimensions=[True, False])
        right = SoftTensorTypeModel([5,10,3], soft_by_dimensions=[True, False, True])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, combined_softness = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 3)
        self.assertEqual([5,10,3], combined_dims)
        self.assertEqual(len(combined_softness), 3)
        self.assertEqual([True, False, True], combined_softness)

    def testCombineSoftLongPartial(self):
        left = SoftTensorTypeModel([5,10], soft_by_dimensions=[True, False])
        right = SoftTensorTypeModel([5,10,3,10,2,5], soft_by_dimensions=[True, False, True, False, False, True])

        helper = SoftTensorBinaryOperatorHelper()

        combined_dims, combined_softness = helper.get_combine_type_dimensions(left, right, "test_combine")

        self.assertIsNotNone(combined_dims)
        self.assertEqual(len(combined_dims), 6)
        self.assertEqual([5,10,3,10,2,5], combined_dims)
        self.assertEqual(len(combined_softness), 6)
        self.assertEqual([True, False, True, False, False, True], combined_softness)

    def testCombineDimMismatchInPartialGivesException(self):
        left = SoftTensorTypeModel([5,10])
        right = SoftTensorTypeModel([5,12,3])

        helper = SoftTensorBinaryOperatorHelper()

        with self.assertRaises(Exception) as context:
            helper.get_combine_type_dimensions(left, right, "test_combine")