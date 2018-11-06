from Mindblocks.model.tensorflow_placeholder.abstract_tensorflow_placeholder_model import \
    AbstractTensorflowPlaceholderModel
import tensorflow as tf


class SoftTensorTfInputManager(AbstractTensorflowPlaceholderModel):

    def __init__(self, dimensions, soft_by_dimensions, type, reference_name):
        AbstractTensorflowPlaceholderModel.__init__(self, type, reference_name)
        self.type = type
        self.dimensions = dimensions
        self.soft_by_dimensions = soft_by_dimensions
        self.name = reference_name

    def __initialize_placeholders__(self):
        feed_dims = self.dimensions

        for idx, dim_is_soft in enumerate(self.soft_by_dimensions):
            if dim_is_soft:
                feed_dims[idx] = None

        placeholder = tf.placeholder(self.get_tensorflow_type(),
                                     feed_dims,
                                     name=self.name)

        placeholders = [placeholder]

        for idx, dim_is_soft in enumerate(self.soft_by_dimensions):
            if dim_is_soft:
                prefix = self.dimensions[:idx]
                length_placeholder = tf.placeholder(tf.int32,
                                     prefix,
                                     name=self.name + "_soft_length_" + str(idx))

                placeholders.append(length_placeholder)

        return placeholders

    def assign_placeholders(self, value):
        data = self.get_placeholders()[0]
        lengths = self.get_placeholders()[1:]

        assign_lengths = [None]*len(self.dimensions)

        pointer = 0
        for i in range(len(assign_lengths)):
            if self.soft_by_dimensions[i]:
                assign_lengths[i] = lengths[pointer]
                pointer += 1

        value.assign(data, length_list=assign_lengths)

    def format_for_input(self, value):
        tf_input = [value.tensor]

        for length in value.get_lengths():
            if length is not None:
                tf_input.append(length)

        return tf_input