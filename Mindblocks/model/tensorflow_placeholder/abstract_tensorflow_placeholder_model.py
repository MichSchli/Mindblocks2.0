import tensorflow as tf


class AbstractTensorflowPlaceholderModel:

    cached_placeholders = None
    name = None
    type = None

    def __init__(self, type, name=None):
        self.name = name
        self.type = type

    def get_tensorflow_type(self):
        tf_type = None
        if self.type == "int":
            tf_type = tf.int32
        elif self.type == "float":
            tf_type = tf.float32
        elif self.type == "bool":
            tf_type = tf.bool
        elif self.type == "string":
            tf_type = tf.string

        return tf_type

    def get_placeholders(self):
        if self.cached_placeholders is None:
            self.cached_placeholders = self.__initialize_placeholders__()

        return self.cached_placeholders
