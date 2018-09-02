import tensorflow as tf
import numpy as np


class ListBatchValueModel:

    item = None
    lengths = None

    def __init__(self, type, item_shape):
        self.type = type
        self.item_shape = item_shape

    def assign(self, sequence_batch, language="python"):
        self.item = sequence_batch

        if language == "python":
            self.lengths = [len(s) for s in sequence_batch]
            self.batch_size = len(sequence_batch)
        else:
            pass

    def get_lengths(self):
        return self.lengths

    def assign_with_lengths(self, sequence_batch, length_batch, language="tensorflow"):
        self.item = sequence_batch
        self.lengths = length_batch
        if language == "tensorflow":
            self.batch_size = tf.shape(length_batch)[0]
        else:
            self.batch_size = len(length_batch)

    def get_tensorflow_output_tensors(self):
        return [self.item, self.lengths]

    def get_batch_size(self):
        return self.batch_size

    def get_maximum_length(self):
        return self.maximum_length

    def get_numpy_type(self):
        np_type = None
        if self.type == "int":
            np_type = np.int32
        elif self.type == "float":
            np_type = np.float32
        return np_type

    def format_for_tensorflow_input(self):
        shape = [self.get_batch_size(), self.get_maximum_length()] + self.item_shape
        seq_input = np.zeros(shape, dtype=self.get_numpy_type())

        for i in range(len(self.item)):
            for j in range(len(self.item[i])):
                seq_input[i][j] = self.item[i][j]

        return [seq_input, self.lengths]

    def apply_dropouts(self, dropout_rate):
        keep_prob = 1 - float(dropout_rate)
        self.item = tf.nn.dropout(self.item, keep_prob=keep_prob)

    def get_items(self):
        return self.item

    def get_lengths(self):
        return self.lengths

    def get_value(self):
        return self.item

    def is_value_type(self, test_type):
        return test_type == "list"

    def cast(self, type):
        for i in range(len(self.lengths)):
            for j in range(len(self.item[i])):
                if type == "bool":
                    self.item[i][j] = self.item[i][j] == "True"

        self.type = "bool"
        return self