from model.component_type.component_type_model import ComponentTypeModel


class Constant(ComponentTypeModel):

    name = "Constant"
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        return ConstantValue(value_dictionary["value"], value_dictionary["type"])

    def execute(self, input_dictionary, value):
        return {"output": value.value}

    def infer_types(self, input_types, value):
        return {"output": value.value_type}

    def infer_dims(self, input_dims, value):
        return {"output": 1}

class ConstantValue:

    value = None
    value_type = None

    def __init__(self, value, value_type):
        if value_type == "float":
            self.value = float(value)
            self.value_type = value_type