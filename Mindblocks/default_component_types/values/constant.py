from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.old.tensor_type import TensorType
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import numpy as np


class Constant(ComponentTypeModel):

    name = "Constant"
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        return ConstantValue(value_dictionary["value"][0][0],
                             value_dictionary["type"][0][0])

    def execute(self, input_dictionary, value, output_value_models, mode):
        output_value_models["output"].assign(value.value)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"output": TensorTypeModel(value.value_type, [] if not value.tensor else [v for v in value.value.shape])}

class ConstantValue(ExecutionComponentValueModel):

    value = None
    value_type = None
    tensor = False

    def __init__(self, value, value_type, tensor=False):
        if tensor and value_type == "float":
            self.value = np.array(value, dtype=np.float32)
            self.value_type = value_type
            self.tensor = True
        elif value_type == "float":
            self.value = float(value)
            self.value_type = value_type
        elif tensor and value_type == "int":
            self.value = np.array(value, dtype=np.int32)
            self.value_type = value_type
            self.tensor = True
        elif value_type == "int":
            self.value = int(value)
            self.value_type = value_type