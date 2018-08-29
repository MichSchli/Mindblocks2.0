from tensorflow.python.framework import dtypes

from Mindblocks.helpers.neural_network.MlpHelper import MlpHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np


class AttentionComponent(ComponentTypeModel):

    name = "Attention"
    in_sockets = ["sequence", "key"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = AttentionValue()
        value.set_scoring_type(value_dictionary["scoring"][0][0])

        if "output_dim" in value_dictionary:
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
        if value.scoring_type == "bilinear":
            transformed_key = value.key_transform.transform(key, mode)
            transformed_key = tf.expand_dims(transformed_key, 1)
            scores = transformed_key * sequence_tensor

            sequence_shape = tf.shape(sequence_tensor)

            head_scores = tf.reshape(scores, [sequence_shape[0], sequence_shape[1], value.attention_heads, -1])

        head_scores = tf.reduce_sum(head_scores, -1)

        minus_inifinity = dtypes.as_dtype(tf.float32).as_numpy_dtype(-np.inf)
        head_scores = self.mask_attention_logits(head_scores, lengths, minus_inifinity)
        attention_weights = tf.nn.softmax(head_scores, 1)
        attention_weights = tf.expand_dims(attention_weights, -1)

        exp_seq_tensor = tf.expand_dims(sequence_tensor, 2)
        attention_weighted_matrix = exp_seq_tensor * attention_weights
        weighted_sums = tf.reduce_sum(attention_weighted_matrix, 1)

        output = tf.reshape(weighted_sums, [sequence_shape[0], sequence_shape[-1]])

        if value.should_combine_context():
            output = tf.nn.relu(tf.concat([output, key], -1))
            output = value.output_transform.transform(output, mode=mode)
            output = tf.nn.tanh(output)
            #if mode == "train":
            #    output = tf.nn.dropout(output, keep_prob=0.7)

        return output

    def mask_attention_logits(self, attention_logits, lengths, score_mask_value):
        seq_mask = tf.sequence_mask(
            lengths,
            maxlen=tf.shape(attention_logits)[1],
            name="attention_mask"
        )
        seq_mask = tf.tile(tf.expand_dims(seq_mask, -1), [1,1,tf.shape(attention_logits)[-1]])
        mask_values = score_mask_value * tf.ones_like(attention_logits)
        return tf.where(seq_mask, attention_logits, mask_values)

    def build_value_type_model(self, input_types, value):
        output_type = input_types["key"].copy()
        output_type.set_inner_dim(value.output_dimension)

        return {"output": output_type}


class AttentionValue(ExecutionComponentValueModel):

    axis = None
    attention_heads = None
    output_dimension = None
    initialized = None
    scoring_type = None

    def __init__(self):
        self.initialized = False
        self.attention_heads = 1

    def set_output_dimension(self, dim):
        self.output_dimension = dim

    def should_combine_context(self):
        return True

    def set_scoring_type(self, scoring_type):
        self.scoring_type = scoring_type

    def set_attention_heads(self, attention_heads):
        self.attention_heads = attention_heads

    def initialize_transforms(self, key_dim, value_dim):
        if self.scoring_type == "bilinear":
            self.key_transform = MlpHelper([int(key_dim), int(value_dim)], "attention_key_input_transform")
            self.output_transform = MlpHelper([int(value_dim) + int(key_dim), int(self.output_dimension)], "attention_output_transform")

        self.initialized = True