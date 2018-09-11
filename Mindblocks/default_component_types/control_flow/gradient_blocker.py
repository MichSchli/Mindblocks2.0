from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf


class GradientBlocker(ComponentTypeModel):

    name = "GradientBlocker"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return GradientBlockerValue()

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        input = input_dictionary["input"].get_value()
        input = tf.stop_gradient(input)
        output_models["output"].assign(input)
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["input"].copy()}


class GradientBlockerValue(ExecutionComponentValueModel):

    pass