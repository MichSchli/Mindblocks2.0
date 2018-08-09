import tensorflow as tf
import numpy as np

from Mindblocks.model.value_type.tensor.tensor_value_model import TensorValueModel


class SequenceBatchValueModel:

    type = None
    item_shape = None

    batch_size = None
    max_length = None

    sequences = None
    sequence_lengths = None

    def __init__(self, type, item_shape):
        self.type = type
        self.item_shape = item_shape[:]

        self.sequences = []
        self.sequence_lengths = []

    def get_sequences(self):
        return self.sequences

    def get_sequence_lengths(self):
        return self.sequence_lengths

    def assign(self, sequence_batch, language="python"):
        self.sequences = sequence_batch

        if language == "python":
            self.sequence_lengths = [len(s) for s in sequence_batch]
            self.batch_size = len(sequence_batch)
            self.max_length = max(self.sequence_lengths)
        else:
            self.sequence_lengths = [20,20]
            self.batch_size = 2
            self.max_length = 20

    def assign_with_lengths(self, sequence_batch, length_batch):
        self.sequences = sequence_batch
        self.sequence_lengths = length_batch

    def get_value(self):
        return self.sequences

    def format_for_tensorflow_input(self):
        shape = [self.get_batch_size(), self.get_max_length()] + self.item_shape
        seq_input = np.zeros(shape, dtype= self.get_numpy_type())

        for i in range(len(self.sequences)):
            for j in range(len(self.sequences[i])):
                seq_input[i][j] = self.sequences[i][j]

        return [seq_input, self.sequence_lengths]

    def get_tensorflow_output_tensors(self):
        return [self.sequences, self.sequence_lengths]

    def get_sequence(self):
        return self.sequences

    def initialize_as_tensorflow_placeholder(self):
        tf_type = self.get_tensorflow_type()
        self.sequences = tf.placeholder(tf_type, shape=[self.get_batch_size(), self.get_max_length()] + self.item_shape)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.get_batch_size()])

    def get_batch_size(self):
        return self.batch_size

    def get_max_length(self):
        return self.max_length

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

    def is_value_type(self, test_type):
        return test_type == "sequence"

    def get_token(self, batch, index):
        token = self.sequences[batch][index]
        tensor_value_model = TensorValueModel(self.type, self.item_shape)
        tensor_value_model.assign(token)
        return tensor_value_model