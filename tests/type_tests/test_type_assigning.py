import unittest

from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class TestTypeUsage(unittest.TestCase):

    def testAssignScalarTensorType(self):
        type = TensorTypeModel("float", [])
        value = type.initialize_value_model()
        value.assign(5.67)

        self.assertEqual(5.67, value.get_value())

    def testAssignScalarBatchSequenceType(self):
        type = SequenceBatchTypeModel("float", [])
        value = type.initialize_value_model()
        value.assign([[5.67, 2.3], [1.2, 1.3, 1.4]])

        self.assertListEqual([[5.67, 2.3], [1.2, 1.3, 1.4]], value.get_sequences())
        self.assertListEqual([2,3], value.get_sequence_lengths())