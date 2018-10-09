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

            if input_dictionary["input"].is_value_type("list"):
                lengths = input_dictionary["input"].lengths
                max_length = tf.reduce_max(lengths)
                mask = tf.sequence_mask(lengths,
                                        maxlen=max_length,
                                        dtype=tf.bool)
                v = v[:, :max_length]

                replacement = tf.zeros_like(v)
                v = tf.where(mask, v, replacement)

            input_dictionary["input"].assign(v, language="tensorflow")
        return {"output": input_dictionary["input"]}

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["input"].copy()}


class LogValue(ExecutionComponentValueModel):

    pass