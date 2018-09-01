from Mindblocks.model.tensorflow_placeholder.abstract_tensorflow_placeholder_model import \
    AbstractTensorflowPlaceholderModel
import tensorflow as tf


class TensorPlaceholderModel(AbstractTensorflowPlaceholderModel):

    def __init__(self, dimensions, type, name=None):
        AbstractTensorflowPlaceholderModel.__init__(self, type, name)
        self.type = type
        self.dimensions = dimensions

    def __initialize_placeholders__(self):
        placeholder = tf.placeholder(self.get_tensorflow_type(),
                                     self.dimensions,
                                     name=self.name)
        return [placeholder]