import unittest

import numpy as np

from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class TestSoftTensorsToNumpy(unittest.TestCase):

    def testScalarToNumpy(self):
        tensor_type = SoftTensorTypeModel([])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign(5)

        self.assertEqual(np.array(5), tensor_value.get_value())
        self.assertEqual([], tensor_value.get_dimensions())
        self.assertEqual([], tensor_value.get_lengths())

    def testListToNumpy(self):
        tensor_type = SoftTensorTypeModel([None])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([5,5, 2,3])

        self.numpyAssertEqual(np.array([5,5, 2,3], dtype=np.float32), tensor_value.get_value())

        self.assertEqual([None], tensor_value.get_dimensions())
        self.assertEqual([None], tensor_value.get_lengths())

    def testArrayToNumpy(self):
        tensor_type = SoftTensorTypeModel([2,2])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2,3]])

        self.numpyAssertEqual(np.array([[5,5],[2,3]], dtype=np.float32), tensor_value.get_value())

        self.assertEqual([2,2], tensor_value.get_dimensions())
        self.assertEqual([None, None], tensor_value.get_lengths())

    def testArrayToNumpyInt(self):
        tensor_type = SoftTensorTypeModel([2,2], string_type="int")
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2,3]])

        self.numpyAssertEqual(np.array([[5,5],[2,3]], dtype=np.int32), tensor_value.get_value())

        self.assertEqual([2,2], tensor_value.get_dimensions())
        self.assertEqual([None, None], tensor_value.get_lengths())

    def testArrayToNumpyString(self):
        tensor_type = SoftTensorTypeModel([2,2], string_type="string")
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([["foo", "bar"],["baz", "tes"]])

        self.numpyAssertEqual(np.array([["foo", "bar"],["baz", "tes"]], dtype=np.object), tensor_value.get_value())

        self.assertEqual([2,2], tensor_value.get_dimensions())
        self.assertEqual([None, None], tensor_value.get_lengths())

    def testSoftArrayToNumpy(self):
        tensor_type = SoftTensorTypeModel([3,4], soft_by_dimensions=[False, True])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2], [1,2,3,4]])

        self.numpyAssertEqual(np.array([[5,5,0,0],[2,0,0,0], [1,2,3,4]], dtype=np.float32), tensor_value.get_value())

        lengths = tensor_value.get_lengths()
        self.assertIsNone(lengths[0])
        self.numpyAssertEqual(np.array([2,1,4], dtype=np.int32), lengths[1])

        self.assertEqual([3,4], tensor_value.get_dimensions())
        self.assertEqual([3,4], tensor_value.get_max_lengths())

    def testSoftArrayToNumpyChopsSize(self):
        tensor_type = SoftTensorTypeModel([3,10], soft_by_dimensions=[False, True])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2], [1,2,3,4]])

        self.numpyAssertEqual(np.array([[5,5,0,0],[2,0,0,0], [1,2,3,4]], dtype=np.float32), tensor_value.get_value())

        lengths = tensor_value.get_lengths()
        self.assertIsNone(lengths[0])
        self.numpyAssertEqual(np.array([2,1,4], dtype=np.int32), lengths[1])

        self.assertEqual([3,10], tensor_value.get_dimensions())
        self.assertEqual([3,4], tensor_value.get_max_lengths())

    def testSoftArrayToNumpyInt(self):
        tensor_type = SoftTensorTypeModel([3,4], string_type="int", soft_by_dimensions=[False, True])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2], [1,2,3,4]])

        self.numpyAssertEqual(np.array([[5,5,0,0],[2,0,0,0], [1,2,3,4]], dtype=np.int32), tensor_value.get_value())

        lengths = tensor_value.get_lengths()
        self.assertIsNone(lengths[0])
        self.numpyAssertEqual(np.array([2,1,4], dtype=np.int32), lengths[1])

        self.assertEqual([3,4], tensor_value.get_dimensions())
        self.assertEqual([3,4], tensor_value.get_max_lengths())

    def testSoftArrayToNumpyInitialDimUnknown(self):
        tensor_type = SoftTensorTypeModel([None,None], soft_by_dimensions=[False, True])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2], [1,2,3,4]])

        self.numpyAssertEqual(np.array([[5,5,0,0],[2,0,0,0], [1,2,3,4]], dtype=np.float32), tensor_value.get_value())

        lengths = tensor_value.get_lengths()
        self.assertIsNone(lengths[0])
        self.numpyAssertEqual(np.array([2,1,4], dtype=np.int32), lengths[1])

        self.assertEqual([None,None], tensor_value.get_dimensions())
        self.assertEqual([3,4], tensor_value.get_max_lengths())

    def testSoftArrayToNumpyInitialDimPartiallyKnown(self):
        tensor_type = SoftTensorTypeModel([None,4], soft_by_dimensions=[False, True])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2], [1,2,3,4]])

        self.numpyAssertEqual(np.array([[5,5,0,0],[2,0,0,0], [1,2,3,4]], dtype=np.float32), tensor_value.get_value())

        lengths = tensor_value.get_lengths()
        self.assertIsNone(lengths[0])
        self.numpyAssertEqual(np.array([2,1,4], dtype=np.int32), lengths[1])

        self.assertEqual([None,4], tensor_value.get_dimensions())
        self.assertEqual([3,4], tensor_value.get_max_lengths())

    def testSoftArrayToNumpyInitialDimTooLarge(self):
        tensor_type = SoftTensorTypeModel([None,10], soft_by_dimensions=[False, True])
        tensor_value = tensor_type.initialize_value_model(language="python")

        tensor_value.initial_assign([[5,5],[2], [1,2,3,4]])

        self.numpyAssertEqual(np.array([[5,5,0,0],[2,0,0,0], [1,2,3,4]], dtype=np.float32), tensor_value.get_value())

        lengths = tensor_value.get_lengths()
        self.assertIsNone(lengths[0])
        self.numpyAssertEqual(np.array([2,1,4], dtype=np.int32), lengths[1])

        self.assertEqual([None,10], tensor_value.get_dimensions())
        self.assertEqual([3,4], tensor_value.get_max_lengths())

    def numpyAssertEqual(self, reference, test):
        reference = reference.flatten()
        test = test.flatten()

        self.assertEqual(reference.dtype, test.dtype)

        for i in range(reference.shape[0]):
            self.assertEqual(reference[i], test[i])