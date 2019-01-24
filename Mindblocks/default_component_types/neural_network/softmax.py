from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Softmax(ComponentTypeModel):

    name = "Softmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return SoftmaxValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        v = input_dictionary["input"].get_value()
        all_lengths = input_dictionary["input"].get_lengths()

        v = tf.nn.softmax(v)

        replacement = tf.zeros_like(v)
        sth = SoftTensorHelper()
        replaced_v = sth.replace_elements_outside_lengths(v, all_lengths, replacement)

        output_value_models["output"].assign(replaced_v, length_list=all_lengths)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()
        return {"output": output_type}

class SoftmaxValue(ExecutionComponentValueModel):

    pass