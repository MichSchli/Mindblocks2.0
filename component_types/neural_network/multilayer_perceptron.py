from model.component.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf
import numpy as np
from model.component.component_value_model import ComponentValueModel


class MultilayerPerceptron(ComponentTypeModel):

    name = "MultilayerPerceptron"
    in_socket_names = ["input"]
    out_socket_names = ["output"]
    available_languages = ["tensorflow"]

    def __init__(self):
        pass

    def get_new_value(self):
        return MultilayerPerceptronValue()

    def execute(self, in_sockets, value, language="python"):
        return [value.transform(in_sockets[0])]

class MultilayerPerceptronValue(ComponentValueModel):

    dims = None
    weights = None
    biases = None

    def __init__(self):
        self.dims = []

    def load(self, value_lines):
        if "dimensions" in value_lines:
            self.dims = [int(d) for d in value_lines["dimensions"][0][0].split(",")]

    def initialize(self):
        self.weights = [None] * (len(self.dims) - 1)
        self.biases = [None] * (len(self.dims) - 1)
        for i in range(len(self.dims) - 1):
            dim_1 = self.dims[i]
            dim_2 = self.dims[i + 1]

            glorot_variance = np.sqrt(6) / np.sqrt(dim_1 + dim_2)
            weight_initializer = np.random.uniform(-glorot_variance, glorot_variance, size=(dim_1, dim_2)).astype(
                np.float32)
            bias_initializer = np.zeros(dim_2, dtype=np.float32)

            self.weights[i] = tf.Variable(weight_initializer, name="W" + str(i))
            self.biases[i] = tf.Variable(bias_initializer, name="b" + str(i))

    def transform(self, vectors):
        for i in range(len(self.dims) - 1):
            vectors = tf.matmul(vectors, self.weights[i]) + self.biases[i]
            if i < len(self.dims) - 2:
                vectors = tf.nn.relu(vectors)

        return vectors

    def copy(self):
        copy = MultilayerPerceptronValue()
        copy.dims = self.dims
        return copy

    def describe(self):
        return "dims=\""+str(self.dims)+"\""