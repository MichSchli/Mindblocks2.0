from Mindblocks.helpers.neural_network.MlpHelper import MlpHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np


class KeyValueAttentionComponent(ComponentTypeModel):

    name = "KeyValueAttention"
    in_sockets = ["sequence", "key"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = KeyValueAttentionValue()
        value.set_output_dimension(int(value_dictionary["output_dim"][0][0]))

        if "heads" in value_dictionary:
            value.set_attention_heads(int(value_dictionary["heads"][0][0]))

        return value

    def execute(self, input_dictionary, value, output_value_models, mode):
        if not value.initialized:
            key_dim = input_dictionary["key"].get_inner_dim()
            value_dim = input_dictionary["sequence"].get_inner_dim()
            value.initialize_transforms(key_dim, value_dim)

        lengths = input_dictionary["sequence"].get_sequence_lengths()

        input_dimension = input_dictionary["sequence"].get_inner_dim()
        attention_result = self.attend(input_dictionary["key"].get_value(),
                                       input_dictionary["sequence"].get_value(),
                                       lengths,
                                       value,
                                       input_dimension,
                                       mode)
        output_value_models["output"].assign(attention_result, language="tensorflow")

        return output_value_models

    def attend(self, key, sequence_tensor, lengths, value, input_dimension, mode):
        key_tensor, value_tensor = tf.split(sequence_tensor, [int(0.5 * input_dimension), int(0.5 * input_dimension)], 2)

        previous_shape = tf.shape(key_tensor)
        transformed_key = tf.reshape(key_tensor, [previous_shape[0] * previous_shape[1], -1])
        transformed_value = value.value_transform.transform(tf.reshape(value_tensor, [previous_shape[0] * previous_shape[1], -1]), mode)

        dim = int(0.5 * input_dimension / value.attention_heads)
        transformed_key = tf.reshape(transformed_key, [previous_shape[0], previous_shape[1], value.attention_heads, dim])
        transformed_value = tf.reshape(transformed_value, [previous_shape[0], previous_shape[1], value.attention_heads, dim])

        transformed_context_key = value.key_input_transform.transform(key, mode)
        transformed_key *= tf.reshape(transformed_context_key, [previous_shape[0], 1, value.attention_heads, dim])

        norm_factor = np.sqrt(dim)
        attention_logits = tf.reduce_sum(transformed_key, axis=3) / norm_factor
        attention_weights = tf.nn.softmax(attention_logits, dim=1)

        attention_weights = self.mask_attention_weights(attention_weights, lengths)

        attention_weights = tf.expand_dims(attention_weights, 3)

        attention_weighted_matrix = transformed_value * attention_weights

        weighted_value_matrix = tf.reduce_sum(attention_weighted_matrix, 1)
        return_value = tf.reshape(weighted_value_matrix, [previous_shape[0], value.output_dimension])

        return return_value

    def mask_attention_weights(self, attention_weights, lengths):
        seq_mask = tf.sequence_mask(
            lengths,
            maxlen=tf.shape(attention_weights)[1],
            dtype=tf.float32,
            name="attention_mask"
        )
        seq_mask = tf.expand_dims(seq_mask, -1)
        attention_weights = attention_weights * seq_mask
        attention_weights = attention_weights / tf.expand_dims(tf.reduce_sum(attention_weights, axis=1), 1)
        return attention_weights

    def build_value_type_model(self, input_types, value):
        output_type = input_types["key"].copy()
        output_type.set_inner_dim(value.output_dimension)

        return {"output": output_type}


class KeyValueAttentionValue(ExecutionComponentValueModel):

    axis = None
    attention_heads = None
    output_dimension = None
    initialized = None

    def __init__(self):
        self.initialized = False
        self.attention_heads = 1

    def set_output_dimension(self, dim):
        self.output_dimension = dim

    def set_attention_heads(self, attention_heads):
        self.attention_heads = attention_heads

    def initialize_transforms(self, key_dim, value_dim):
        self.key_input_transform = MlpHelper([int(key_dim), self.output_dimension], "attention_key_input_transform")
        self.value_transform = MlpHelper([int(value_dim/2), self.output_dimension], "attention_value_transform")
        self.initialized = True