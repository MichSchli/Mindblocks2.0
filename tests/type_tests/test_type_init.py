import unittest
import numpy as np

from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class TestTypeInit(unittest.TestCase):

    """
    Tensors:
    """
    def testScalarTensorTypeInitReturnsCorrectValueInt(self):
        type_model = TensorTypeModel("int", [])
        value_model = type_model.initialize_value_model()
        value_model.initialize_value()

        self.assertEqual("int", value_model.type)
        self.assertEqual(0, value_model.get_value())
        self.assertEqual([], value_model.dimensions)

    def testVectorTensorTypeInitReturnsCorrectValueInt(self):
        type_model = TensorTypeModel("int", [5])
        value_model = type_model.initialize_value_model()
        value_model.initialize_value()

        self.assertEqual("int", value_model.type)
        self.assertTrue(np.allclose(np.zeros(5, dtype=np.int32), value_model.get_value()))
        self.assertEqual([5], value_model.dimensions)

    def testMatrixTensorTypeInitReturnsCorrectValueInt(self):
        type_model = TensorTypeModel("int", [5, 7])
        value_model = type_model.initialize_value_model()
        value_model.initialize_value()

        self.assertEqual("int", value_model.type)
        self.assertTrue(np.allclose(np.zeros([5,7], dtype=np.int32), value_model.get_value()))
        self.assertEqual([5, 7], value_model.dimensions)

    def testScalarTensorTypeInitReturnsCorrectValueFloat(self):
        type_model = TensorTypeModel("float", [])
        value_model = type_model.initialize_value_model()
        value_model.initialize_value()

        self.assertEqual("float", value_model.type)
        self.assertEqual(0, value_model.get_value())
        self.assertEqual([], value_model.dimensions)

    def testVectorTensorTypeInitReturnsCorrectValueFloat(self):
        type_model = TensorTypeModel("float", [5])
        value_model = type_model.initialize_value_model()
        value_model.initialize_value()

        self.assertEqual("float", value_model.type)
        self.assertTrue(np.allclose(np.zeros(5, dtype=np.float32), value_model.get_value()))
        self.assertEqual([5], value_model.dimensions)

    def testMatrixTensorTypeInitReturnsCorrectValueFloat(self):
        type_model = TensorTypeModel("float", [5, 7])
        value_model = type_model.initialize_value_model()
        value_model.initialize_value()

        self.assertEqual("float", value_model.type)
        self.assertTrue(np.allclose(np.zeros([5,7], dtype=np.float32), value_model.get_value()))
        self.assertEqual([5, 7], value_model.dimensions)

    """
    Sequences:
    """
    def testScalarSequenceBatchTypeInitReturnsCorrectValueInt(self):
        type_model = SequenceBatchTypeModel("int", [], None, None)
        value_model = type_model.initialize_value_model()

        self.assertEqual("int", value_model.type)
        self.assertEqual([], value_model.get_sequences())
        self.assertEqual([], value_model.get_sequence_lengths())

    def testScalarSequenceBatchTypeInitReturnsCorrectValueFloat(self):
        type_model = SequenceBatchTypeModel("float", [], None, None)
        value_model = type_model.initialize_value_model()

        self.assertEqual("float", value_model.type)
        self.assertEqual([], value_model.get_sequences())
        self.assertEqual([], value_model.get_sequence_lengths())