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

    def execute(self, input_dictionary, value, output_value_models, mode):
        argmax = np.argmax(input_dictionary["input"].get_value(), axis=-1)
        output_value_models["output"].assign(argmax)

        return output_value_models

    def build_value_type_model(self, input_types, value):
        output_type = input_types["input"].copy()
        output_type.set_inner_dim(1)
        return {"output": output_type}

class ArgmaxValue(ExecutionComponentValueModel):

    pass