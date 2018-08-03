from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Constant(ComponentTypeModel):

    name = "Constant"
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        return ConstantValue(value_dictionary["value"][0],
                             value_dictionary["type"][0])

    def execute(self, input_dictionary, value, mode):
        return {"output": value.value}

    def infer_types(self, input_types, value):
        return {"output": value.value_type}

    def infer_dims(self, input_dims, value):
        return {"output": 1}

class ConstantValue(ExecutionComponentValueModel):

    value = None
    value_type = None

    def __init__(self, value, value_type):
        if value_type == "float":
            self.value = float(value)
            self.value_type = value_type