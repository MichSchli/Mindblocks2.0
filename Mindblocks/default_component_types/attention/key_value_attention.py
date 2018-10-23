from tensorflow.python.framework import dtypes

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

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        if not value.initialized:
            key_dim = input_dictionary["key"].get_dimension(-1)
            value_dim = input_dictionary["sequence"].get_dimension(-1)
            value.initialize_transforms(key_dim, value_dim)

        lengths = input_dictionary["sequence"].get_lengths()[1]

        input_dimension = input_dictionary["sequence"].get_dimension(-1)
        attention_result = self.attend(input_dictionary["key"].get_value(),
                                       input_dictionary["sequence"].get_value(),
                                       lengths,
                                       value,
                                       input_dimension,
                                       mode)
        output_value_models["output"].assign(attention_result, length_list=None)

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

        minus_inifinity = dtypes.as_dtype(tf.float32).as_numpy_dtype(-np.inf)
        self.mask_attention_logits(attention_logits, lengths, minus_inifinity)
        attention_weights = tf.nn.softmax(attention_logits, dim=1)

        #attention_weights = self.mask_attention_weights(attention_weights, lengths)

        attention_weights = tf.expand_dims(attention_weights, 3)

        attention_weighted_matrix = transformed_value * attention_weights

        weighted_value_matrix = tf.reduce_sum(attention_weighted_matrix, 1)
        return_value = tf.reshape(weighted_value_matrix, [previous_shape[0], value.output_dimension])

        return return_value

    def mask_attention_logits(self, attention_logits, lengths, score_mask_value):
        seq_mask = tf.sequence_mask(
            lengths,
            maxlen=tf.shape(attention_logits)[1],
            name="attention_mask"
        )
        seq_mask = tf.tile(tf.expand_dims(seq_mask, -1), [1,1,tf.shape(attention_logits)[-1]])
        mask_values = score_mask_value * tf.ones_like(attention_logits)
        return tf.where(seq_mask, attention_logits, mask_values)

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["key"].copy()
        output_type.set_dimension(-1, value.output_dimension)

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