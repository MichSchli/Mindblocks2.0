from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf
import numpy as np

class LayerNorm(ComponentTypeModel):

    name = "LayerNorm"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return LayerNormValue()

    def execute(self, execution_component, input_dictionary, execution_value, output_models, mode):
        value = input_dictionary["input"].get_value()

        per_batch_mus = tf.reduce_mean(value, axis=-1)
        shifted_value = value - tf.expand_dims(per_batch_mus, -1)

        per_batch_sigmas = tf.sqrt(tf.reduce_mean(tf.square(shifted_value), axis=-1) + 1e-12)
        scaled_value = shifted_value / (tf.expand_dims(per_batch_sigmas, -1))

        output_value = scaled_value * execution_value.get_scaling() + execution_value.get_bias()

        output_models["output"].assign(output_value)
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        inner_dim = input_types["input"].get_inner_dim()
        value.set_inner_dim(inner_dim)
        return {"output": input_types["input"].copy()}


class LayerNormValue(ExecutionComponentValueModel):

    dim = None
    bias = None
    scaling = None

    def set_inner_dim(self, dim):
        self.dim = dim
        self.initialize()

    def initialize(self):
        self.bias = tf.Variable(np.zeros(self.dim, np.float32), dtype=tf.float32)
        self.scaling = tf.Variable(np.ones(self.dim, np.float32), dtype=tf.float32)

    def get_bias(self):
        return self.bias

    def get_scaling(self):
        return self.scaling