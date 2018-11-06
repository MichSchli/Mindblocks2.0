from Mindblocks.model.tensorflow_placeholder.abstract_tensorflow_placeholder_model import \
    AbstractTensorflowPlaceholderModel
import tensorflow as tf


class SoftTensorTfOutputManager:

    def __init__(self, dimensions, soft_by_dimensions, type):
        self.type = type
        self.dimensions = dimensions
        self.soft_by_dimensions = soft_by_dimensions

    def format_for_output(self, value):
        tf_output = [value.tensor]

        for length in value.get_lengths():
            if length is not None:
                tf_output.append(length)

        return tf_output

    def assign_tensorflow_output(self, value, tensorflow_output):
        lengths = tensorflow_output[1:]
        assign_lengths = [None]*len(self.dimensions)

        pointer = 0
        for i in range(len(assign_lengths)):
            if self.soft_by_dimensions[i]:
                assign_lengths[i] = lengths[pointer]
                pointer += 1

        value.assign(tensorflow_output[0], length_list=assign_lengths)