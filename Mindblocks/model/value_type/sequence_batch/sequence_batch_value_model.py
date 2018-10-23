import tensorflow as tf
import numpy as np

class OldSequenceBatchValueModel:

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

    def get_batch_size(self):
        return self.batch_size

    def get_inner_dim(self):
        return self.item_shape[-1] if len(self.item_shape) > 0 else 1

    def get_maximum_sequence_length(self):
        return self.max_length

    def assign(self, sequence_batch, language="python"):
        self.sequences = sequence_batch

        if language == "python":
            self.sequence_lengths = [len(s) for s in sequence_batch]
            self.batch_size = len(sequence_batch)
            self.max_length = max(self.sequence_lengths)
        else:
            pass

    def assign_with_lengths(self, sequence_batch, length_batch, language="tensorflow"):
        self.sequences = sequence_batch
        self.sequence_lengths = length_batch

    def apply_dropouts(self, dropout_rate, dropout_dim=None):
        keep_prob = 1 - float(dropout_rate)

        noise_shape = None
        if dropout_dim is not None:
            in_shape = tf.shape(self.sequences)
            kept_shape_dims = in_shape[:dropout_dim+1]
            dropped_out_dims = in_shape[dropout_dim+1:]

            noise_shape = tf.concat([kept_shape_dims, tf.ones_like(dropped_out_dims)], axis=-1)

        self.sequences = tf.nn.dropout(self.sequences, keep_prob=keep_prob, noise_shape=noise_shape)
        return self

    def get_value(self):
        return self.sequences

    def format_for_tensorflow_input(self):
        shape = [self.get_batch_size(), self.get_maximum_sequence_length()] + self.item_shape
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
        self.sequences = tf.placeholder(tf_type, shape=[self.get_batch_size(), self.get_maximum_sequence_length()] + self.item_shape)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.get_batch_size()])

    def get_tensorflow_type(self):
        tf_type = None
        if self.type == "int":
            tf_type = tf.int32
        elif self.type == "float":
            tf_type = tf.float32
        elif self.type == "string":
            tf_type = tf.string
        return tf_type

    def get_numpy_type(self):
        np_type = None
        if self.type == "int":
            np_type = np.int32
        elif self.type == "float":
            np_type = np.float32
        elif self.type == "string":
            np_type = np.str
        return np_type

    def is_value_type(self, test_type):
        return test_type == "sequence"

    def get_token(self, batch, index):
        token = self.sequences[batch][index]
        tensor_value_model = TensorValueModel(self.type, self.item_shape)
        tensor_value_model.assign(token)
        return tensor_value_model