from Mindblocks.model.tensorflow_placeholder.abstract_tensorflow_placeholder_model import \
    AbstractTensorflowPlaceholderModel
import tensorflow as tf


class PaddedTensorPlaceholderModel(AbstractTensorflowPlaceholderModel):

    def __init__(self, batch_size, max_length, dimensions, type, name=None):
        AbstractTensorflowPlaceholderModel.__init__(self, type, name)
        self.type = type
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_length = max_length

    def __initialize_placeholders__(self):
        tf_type = self.get_tensorflow_type()
        placeholder = tf.placeholder(tf_type, shape=[self.batch_size, self.max_length] + self.dimensions)
        lengths = tf.placeholder(tf.int32, shape=[self.batch_size])

        return [placeholder, lengths]