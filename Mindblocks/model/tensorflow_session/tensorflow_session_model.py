import tensorflow as tf


class TensorflowSessionModel:

    identifier = None
    tensorflow_session = None

    current_iteration = None
    __tensorflow_iteration__ = None

    def __init__(self):
        self.current_iteration = 0

    def get_session(self):
        return self.tensorflow_session

    def initialize_variables(self):
        self.tensorflow_session.run(tf.global_variables_initializer())

    def run(self, variables, feed_dict):
        tensorflow_iteration = self.__tensorflow_iteration__
        if tensorflow_iteration is not None:
            feed_dict[tensorflow_iteration] = self.current_iteration

        tf_outputs = self.tensorflow_session.run(variables, feed_dict=feed_dict)
        return tf_outputs

    def update_iteration(self, iteration):
        self.current_iteration = iteration

    def get_tensorflow_iteration(self):
        if self.__tensorflow_iteration__ is None:
            self.__tensorflow_iteration__ = tf.placeholder(tf.int32, shape=[])
        return self.__tensorflow_iteration__