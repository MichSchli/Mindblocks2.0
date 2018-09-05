from Mindblocks.helpers.neural_network.MlpHelper import MlpHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf
import numpy as np

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class MultilayerPerceptron(ComponentTypeModel):

    name = "MultilayerPerceptron"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        if "dropout_rate" in value_dictionary:
            dropout_rate = float(value_dictionary["dropout_rate"][0][0])
        else:
            dropout_rate = 0.0

        v = MultilayerPerceptronValue([int(d) for d in value_dictionary["dimensions"][0][0].split(",")],
                                         dropout_rate=dropout_rate)
        v.language = language
        return v

    def execute(self, input_dictionary, value, output_value_models, mode):
        in_value = input_dictionary["input"].get_value()
        in_shape = tf.shape(in_value)

        to_mlp_value = tf.reshape(in_value, [-1, in_shape[-1]])
        post_value = value.transform(to_mlp_value, mode)

        if value.dims[-1] == 1:
            out_shape = in_shape[:-1]
        else:
            out_shape = tf.concat([in_shape[:-1], [value.dims[-1]]], axis=-1)
        post_value = tf.reshape(post_value, out_shape)

        if input_dictionary["input"].is_value_type("list"):
            lengths = input_dictionary["input"].lengths
            output_value_models["output"].assign_with_lengths(post_value, lengths, language=value.language)
        else:
            output_value_models["output"].assign(post_value, language=value.language)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()
        output_type.set_inner_dim(value.dims[-1])
        return {"output": output_type}


class MultilayerPerceptronValue(ExecutionComponentValueModel):

    dims = None
    weights = None
    biases = None
    dropout_rate = None

    def __init__(self, dims, dropout_rate=0.0):
        self.dims = dims
        self.dropout_rate = dropout_rate

        self.mlp_helper = MlpHelper(dims, variable_prefix=self.get_name(), dropout_rate=dropout_rate)

    def count_parameters(self):
        return self.mlp_helper.count_parameters()

    def transform(self, vectors, mode):
        return self.mlp_helper.transform(vectors, mode)