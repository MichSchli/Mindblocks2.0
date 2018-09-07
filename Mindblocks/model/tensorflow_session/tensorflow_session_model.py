import os

import tensorflow as tf
from tensorflow.python.client import timeline

from Mindblocks.model.abstract.abstract_model import AbstractModel


class TensorflowSessionModel(AbstractModel):

    identifier = None
    tensorflow_session = None
    should_profile = False

    current_iteration = None
    __tensorflow_iteration__ = None

    tf_run_options = None
    tf_run_metadata = None

    saver = None

    def __init__(self):
        self.current_iteration = 0

    def get_session(self):
        return self.tensorflow_session

    def initialize_variables(self):
        self.tensorflow_session.run(tf.global_variables_initializer())

    def generate_profiling_data(self, log_dir):
        fetched_timeline = timeline.Timeline(self.tf_run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        writer = tf.summary.FileWriter(logdir=log_dir + '/log', graph=self.tensorflow_session.graph)
        with open(log_dir + '/timeline.json', 'w') as f:
            f.write(chrome_trace)

        writer.add_run_metadata(self.tf_run_metadata, "mySess")
        writer.close()

    def run(self, variables, feed_dict):
        if self.should_profile and self.tf_run_options is None:
            self.tf_run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.tf_run_metadata = tf.RunMetadata()

        tensorflow_iteration = self.__tensorflow_iteration__
        if tensorflow_iteration is not None:
            feed_dict[tensorflow_iteration] = self.current_iteration

        tf_outputs = self.tensorflow_session.run(variables,
                                                 feed_dict=feed_dict,
                                                 options=self.tf_run_options,
                                                 run_metadata=self.tf_run_metadata)

        return tf_outputs

    def update_iteration(self, iteration):
        self.current_iteration = iteration

    def get_tensorflow_iteration(self):
        if self.__tensorflow_iteration__ is None:
            self.__tensorflow_iteration__ = tf.placeholder(tf.int32, shape=[], name="iteration_number")
        return self.__tensorflow_iteration__

    def save(self, filepath):
        if self.saver is None:
            self.saver = tf.train.Saver()

        save_dir = os.path.dirname(filepath)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.saver.save(self.tensorflow_session, filepath)

    def load(self, filepath):
        if self.saver is None:
            self.saver = tf.train.Saver()

        self.saver.restore(self.tensorflow_session, filepath)