import tensorflow as tf


class ListBatchValueModel:

    item = None
    lengths = None

    def assign(self, sequence_batch, language="python"):
        self.item = sequence_batch

        if language == "python":
            self.lengths = [len(s) for s in sequence_batch]
        else:
            pass

    def assign_with_lengths(self, sequence_batch, length_batch, language="tensorflow"):
        self.item = sequence_batch
        self.lengths = length_batch

    def get_tensorflow_output_tensors(self):
        return [self.item, self.lengths]

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