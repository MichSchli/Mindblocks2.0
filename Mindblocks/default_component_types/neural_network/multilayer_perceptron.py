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

        return MultilayerPerceptronValue([int(d) for d in value_dictionary["dimensions"][0][0].split(",")],
                                         dropout_rate=dropout_rate)

    def execute(self, input_dictionary, value, output_value_models, mode):
        post_value = value.transform(input_dictionary["input"].get_value(), mode)
        output_value_models["output"].assign(post_value)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        output_type = input_types["input"].copy()
        output_type.set_inner_dim(value.dims[-1])
        return {"output": output_type}


class MultilayerPerceptronValue(ExecutionComponentValueModel):

    dims = None
    weights = None
    biases = None
    dropout_rate = None

    variable_prefix = "abc"

    def __init__(self, dims, dropout_rate=0.0):
        self.dims = dims
        self.initialize_weights_and_biases()
        self.dropout_rate = dropout_rate

    def initialize_weights_and_biases(self):
        self.weights = [None] * len(self.dims)
        self.biases = [None] * len(self.dims)

        for i in range(len(self.dims) - 1):
            dim_1 = self.dims[i]
            dim_2 = self.dims[i + 1]

            glorot_variance = np.sqrt(6) / np.sqrt(dim_1 + dim_2)
            weight_initializer = np.random.uniform(-glorot_variance, glorot_variance, size=(dim_1, dim_2)).astype(
                np.float32)
            bias_initializer = np.zeros(dim_2, dtype=np.float32)

            self.weights[i] = tf.Variable(weight_initializer, name=self.variable_prefix + "_W" + str(i))
            self.biases[i] = tf.Variable(bias_initializer, name=self.variable_prefix + "_b" + str(i))

    def transform(self, vectors, mode):
        keep_prob = 1.0 - self.dropout_rate
        for i in range(len(self.dims) - 1):
            vectors = tf.matmul(vectors, self.weights[i]) + self.biases[i]
            if i < len(self.dims) - 2:
                if self.dropout_rate > 0.00001 and mode == "train":
                    vectors = tf.nn.dropout(vectors, keep_prob=keep_prob)

                vectors = tf.nn.relu(vectors)

        return vectors