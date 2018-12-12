from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class Switch(ComponentTypeModel):

    name = "Switch"
    in_sockets = ["left", "right", "switch_input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = SwitchValue()
        value.language = language
        value.switch_input_type = value_dictionary["switch_input_type"][0][0]
        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        #TODO: Proper handling of various cases for this:
        if value.switch_input_type == "logits":
            switch_logits = input_dictionary["switch_input"].get_value()

            #TODO: Hardcoded expand dims
            switch_values = tf.expand_dims(tf.nn.sigmoid(switch_logits), -1)

            all_lengths = input_dictionary["switch_input"].get_lengths()
            replacement = tf.zeros_like(switch_values)
            sth = SoftTensorHelper()
            switch_values = sth.replace_elements_outside_lengths(switch_values, all_lengths, replacement)

        left = input_dictionary["left"].get_value()
        right = input_dictionary["right"].get_value()
        lengths = input_dictionary["left"].get_lengths()

        output = switch_values * left + (1 - switch_values) * right
        output_models["output"].assign(output, length_list=lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["left"].copy()}


class SwitchValue(ExecutionComponentValueModel):

    switch_input_type = None