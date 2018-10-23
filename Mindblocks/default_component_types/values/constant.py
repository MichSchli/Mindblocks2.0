from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel
import numpy as np


class Constant(ComponentTypeModel):

    name = "Constant"
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ConstantValue(value_dictionary["value"][0][0],
                             value_dictionary["type"][0][0])

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        output_value_models["output"].assign(value.value, length_list=None)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):

        output_tensor_type = SoftTensorTypeModel([] if not value.tensor else [v for v in value.value.shape],
                                                 string_type=value.value_type)

        return {"output": output_tensor_type}


class ConstantValue(ExecutionComponentValueModel):

    value = None
    value_type = None
    tensor = False

    def __init__(self, value, value_type, tensor=False):
        if " " in value:
            tensor = True

        if tensor and value_type == "float":
            self.value = np.array([np.fromstring(v, dtype=np.float32, sep=" ") for v in value.split(",")])
            self.value_type = value_type
            self.tensor = True
        elif value_type == "float":
            self.value = float(value)
            self.value_type = value_type
        elif tensor and value_type == "int":
            self.value = np.fromstring(value, dtype=np.int32, sep=" ")
            self.value_type = value_type
            self.tensor = True
        elif value_type == "int":
            self.value = int(value)
            self.value_type = value_type