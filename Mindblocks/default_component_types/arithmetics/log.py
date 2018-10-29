from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class Log(ComponentTypeModel):

    name = "Log"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = LogValue()
        value.language = language
        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.language == "python":
            print("didnt both defining log for python")
            exit()
        elif value.language == "tensorflow":
            v = input_dictionary["input"].get_value()

            v = tf.log(v)
            all_lengths = input_dictionary["input"].get_lengths()
            replacement = tf.zeros_like(v)

            sth = SoftTensorHelper()
            replaced_v = sth.replace_elements_outside_lengths(v, all_lengths, replacement)

            output_models["output"].assign(replaced_v, length_list=all_lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["input"].copy()}


class LogValue(ExecutionComponentValueModel):

    pass