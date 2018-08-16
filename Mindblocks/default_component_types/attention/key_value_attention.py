from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class KeyValueAttentionComponent(ComponentTypeModel):

    name = "KeyValueAttention"
    in_sockets = ["sequence", "key"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return KeyValueAttentionValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        mean = tf.reduce_mean(input_dictionary["sequence"].get_value(), axis=1)
        mean = mean[:, :50]
        output_value_models["output"].assign(mean, language="tensorflow")

        return output_value_models

    def build_value_type_model(self, input_types, value):
        output_type = input_types["key"].copy()

        return {"output": output_type}

class KeyValueAttentionValue(ExecutionComponentValueModel):

    axis = None
    attention_heads = 1

    def __init__(self):
        pass