import tensorflow as tf
import numpy as np


#TODO this should be refactored
class MlpHelper:

    variable_prefix = None

    def __init__(self, dims, variable_prefix, dropout_rate=0.0, use_bias=True):
        self.dims = dims
        self.dropout_rate = dropout_rate
        self.variable_prefix = variable_prefix
        self.use_bias = use_bias

        self.initialize_weights_and_biases()


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

            if self.use_bias:
                self.biases[i] = tf.Variable(bias_initializer, name=self.variable_prefix + "_b" + str(i))

    def transform(self, vectors, mode, activate_output=False):
        keep_prob = 1.0 - self.dropout_rate
        for i in range(len(self.dims) - 1):
            vectors = tf.matmul(vectors, self.weights[i])
            if self.use_bias:
                vectors += self.biases[i]

            if i < len(self.dims) - 2:
                if self.dropout_rate > 0 and mode == "train":
                    vectors = tf.nn.dropout(vectors, keep_prob=keep_prob)

                vectors = tf.nn.relu(vectors)

        if activate_output:
            if self.dropout_rate > 0 and mode == "train":
                vectors = tf.nn.dropout(vectors, keep_prob=keep_prob)
            vectors = tf.nn.relu(vectors)

        return vectors

    def count_parameters(self):
        params = 0
        for i in range(len(self.dims) - 1):
            params += self.dims[i] * self.dims[i + 1]
            if self.use_bias:
                params += self.dims[i + 1]

        return params