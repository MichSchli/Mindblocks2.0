from Mindblocks.model.value_type.old.abstract_value_type import AbstractValueType
import tensorflow as tf


class TensorType(AbstractValueType):

    def __init__(self, type, dims):
        self.type = type
        self.dims = dims

    def format_for_tensorflow_input(self, value):
        return value

    def copy(self):
        return TensorType(self.type, self.dims)

    def get_data_type(self):
        return self.type

    def get_inner_dim(self):
        if len(self.dims) > 0:
            return self.dims[-1]

    def set_inner_dim(self, size):
        if len(self.dims) > 0:
            self.dims[-1] = size

    def get_outer_dim_size(self):
        if len(self.dims) > 0:
            return self.dims[0]

    def set_outer_dim_size(self, size):
        if len(self.dims) > 0:
            self.dims[0] = size

    def get_tensorflow_placeholder(self):
        tf_type = None
        if self.type == "float":
            tf_type = tf.float32
        elif self.type == "int":
            tf_type = tf.int32

        return tf.placeholder(tf_type, self.dims if len(self.dims) > 0 else ())

    def format_from_tensorflow_output(self, value):
        return value