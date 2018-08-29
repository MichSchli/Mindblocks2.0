import numpy as np
import tensorflow as tf


class TensorValueModel:

    type = None
    dimensions = None
    value = None

    def __init__(self, type, dimensions):
        self.type = type
        self.dimensions = [d for d in dimensions]

    def initialize_value(self):
        np_type = self.get_numpy_type()

        self.value = np.zeros(self.dimensions, dtype=np_type)

    def initialize_as_tensorflow_placeholder(self):
        tf_type = self.get_tensorflow_type()
        self.value = tf.placeholder(tf_type, shape=self.dimensions)

    def get_tensorflow_type(self):
        tf_type = None
        if self.type == "int":
            tf_type = tf.int32
        elif self.type == "float":
            tf_type = tf.float32
        return tf_type

    def get_numpy_type(self):
        np_type = None
        if self.type == "int":
            np_type = np.int32
        elif self.type == "float":
            np_type = np.float32
        return np_type

    def get_inner_dim(self):
        return self.dimensions[-1] if len(self.dimensions) > 0 else 1

    def assign(self, value, language=None):
        self.value = value

    def get_value(self):
        return self.value

    def format_for_tensorflow_input(self):
        return [self.value]

    def get_tensorflow_output_tensors(self):
        return [self.value]

    def apply_dropouts(self, dropout_rate):
        keep_prob = 1 - float(dropout_rate)
        self.value = tf.nn.dropout(self.value, keep_prob=keep_prob)

    def is_value_type(self, test_type):
        return test_type == "tensor"

    def cast(self, new_type):
        if new_type == "float":
            new_value_model = TensorValueModel("float", self.dimensions)
            new_value_model.assign(float(self.value))
        if new_type == "int":
            new_value_model = TensorValueModel("int", self.dimensions)
            new_value_model.assign(int(self.value))

        return new_value_model