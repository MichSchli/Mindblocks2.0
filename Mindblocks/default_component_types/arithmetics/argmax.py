from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np

class Argmax(ComponentTypeModel):

    name = "Argmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return ArgmaxValue()

    def execute(self, input_dictionary, value, mode):
        return {"output": np.argmax(input_dictionary["input"], axis=-1)}

    def infer_types(self, input_types, value):
        return {"output": input_types["input"]}

    def infer_dims(self, input_dims, value):
        return {"output": input_dims["input"][:-1]}

class ArgmaxValue:

    pass