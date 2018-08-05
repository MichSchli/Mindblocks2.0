from Mindblocks.model.value_type.abstract_value_type import AbstractValueType
import tensorflow as tf
import numpy as np


class SequenceBatchType(AbstractValueType):

    def __init__(self, type, dim, max_length):
        self.type = type
        self.max_length = max_length
        self.dim = dim

    def copy(self):
        return SequenceBatchType(self.type, self.dim, self.max_length)

    def set_data_type(self, type_str):
        self.type = type_str

    def set_inner_dim(self, dim):
        if self.dim == [] and dim == 1:
            pass
        elif self.dim == []:
            self.dim = [dim]
        else:
            self.dim[-1] = dim

    def extend_dims(self, dim):
        self.dim + [dim]

    def format_for_tensorflow_input(self, value):
        padded_shape = self.get_padded_shape()
        padded_shape[0] = len(value)

        padded = np.zeros(padded_shape, dtype=self.get_numpy_type())
        for i in range(len(value)):
            for j in range(len(value[i])):
                padded[i][j] = np.array(value[i][j], dtype=self.get_numpy_type())

        return padded

    def format_from_tensorflow_output(self, value):
        empty_python_sequence = self.get_python_sequence()

        for i in range(len(empty_python_sequence)):
            for j in range(len(empty_python_sequence[i])):
                empty_python_sequence[i][j] = value[i][j]

        return empty_python_sequence

    def get_python_sequence(self):
        return [[None] * length for length in self.sequence_lengths]

    def get_tensorflow_placeholder(self):
        tf_type = None
        if self.type == "float":
            tf_type = tf.float32
        elif self.type == "int":
            tf_type = tf.int32

        shape = self.get_padded_shape()

        return tf.placeholder(tf_type, shape=shape)

    def get_numpy_type(self):
        if self.type == "float":
            return np.float32
        elif self.type == "int":
            return np.int32

    def get_padded_shape(self):
        if len(self.dim) > 0:
            return [None, self.max_length] + self.dim
        else:
            return [None, self.max_length]