from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Softmax(ComponentTypeModel):

    name = "Softmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return SoftmaxValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        output_value_models["output"].assign(tf.nn.softmax(input_dictionary["input"].get_value()))
        return output_value_models

    def build_value_type_model(self, input_types, value):
        output_type = input_types["input"].copy()
        output_type.set_inner_dim(1)
        return {"output": output_type}

class SoftmaxValue(ExecutionComponentValueModel):

    pass