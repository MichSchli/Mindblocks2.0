from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Concat(ComponentTypeModel):

    name = "Concat"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ConcatValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        if value.new_array:
            result = np.array([input_dictionary["left"].get_value(), input_dictionary["right"].get_value()])
        else:
            result = np.concatenate((input_dictionary["left"].get_value(), input_dictionary["right"].get_value()))
        output_value_models["output"].assign(result)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        left_dims = input_types["left"].get_dimensions()
        right_dims = input_types["right"].get_dimensions()

        if len(left_dims) == 0 and len(right_dims) == 0:
            value.new_array = True

        return {"output": input_types["left"].copy()}


class ConcatValue(ExecutionComponentValueModel):

    new_array = False