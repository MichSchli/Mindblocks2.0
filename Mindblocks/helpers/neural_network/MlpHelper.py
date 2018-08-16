import tensorflow as tf
import numpy as np

#TODO this should be refactored
class MlpHelper:

    variable_prefix = None

    def __init__(self, dims, variable_prefix, dropout_rate=0.0):
        self.dims = dims
        self.dropout_rate = dropout_rate
        self.variable_prefix = variable_prefix
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

            weight_initializer = np.ones_like(weight_initializer)

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