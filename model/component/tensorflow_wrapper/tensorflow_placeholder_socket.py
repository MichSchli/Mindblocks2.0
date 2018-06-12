import tensorflow as tf

class TensorflowPlaceholderSocket:

    value = None
    edge = None

    def __init__(self, edge):
        value_type = edge.get_value_type()

        tf_type = None
        if value_type.type == "float" or value_type.type == "array:float":
            tf_type = tf.float32
        elif value_type.type == "int":
            tf_type = tf.int32

        if value_type.dim == 1:
            shape = None
        else:
            shape = value_type.dim

        self.value = tf.placeholder(shape=shape, dtype=tf_type)
        self.edge = edge

    def get_value(self):
        return self.value