import tensorflow as tf

class TensorflowPlaceholderSocket:

    value = None

    def __init__(self):
        self.value = tf.placeholder(dtype=tf.float32)

    def get_value(self):
        return self.value