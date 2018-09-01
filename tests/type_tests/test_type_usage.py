import unittest
import numpy as np

from Mindblocks.default_component_types.values.constant import ConstantValue
from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
from Mindblocks.repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from Mindblocks.repository.creation_component_repository.creation_component_specifications import \
    CreationComponentSpecifications
from tests.setup_holder import SetupHolder


class TestTypeUsage(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testConstantYieldsScalarTensorType(self):
        spec = ComponentTypeSpecifications()
        spec.name = "Constant"
        constant_type = self.setup_holder.type_repository.get(spec)[0]

        value = ConstantValue("5.67", "float")

        type = constant_type.build_value_type_model({}, value, mode="train")["output"]

        self.assertEqual("float", type.type)
        self.assertEqual([], type.dimensions)

        value = type.initialize_value_model()
        value.assign(5.67)

        self.assertEqual(5.67, value.get_value())

    def testConstantWithMatrixYieldsTensorType(self):
        spec = ComponentTypeSpecifications()
        spec.name = "Constant"
        constant_type = self.setup_holder.type_repository.get(spec)[0]

        value = ConstantValue("5 2.7, 1 20", "float", tensor=True)

        type = constant_type.build_value_type_model({}, value, mode="train")["output"]

        self.assertEqual("float", type.type)
        self.assertEqual([2,2], type.dimensions)

        value = type.initialize_value_model()
        value.assign([[5, 2.7], [1, 20]])

        self.assertTrue(np.allclose(np.array([[5, 2.7], [1, 20]], dtype=np.float32), value.get_value()))