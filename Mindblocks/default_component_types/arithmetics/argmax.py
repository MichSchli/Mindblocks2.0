from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Argmax(ComponentTypeModel):

    name = "Argmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return ArgmaxValue()

    def execute(self, input_dictionary, value, mode):
        return {"output": np.argmax(input_dictionary["input"], axis=-1)}

    def build_value_type(self, input_types, value):
        return {"output": input_types["input"].copy()}

class ArgmaxValue(ExecutionComponentValueModel):

    pass