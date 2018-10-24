import numpy as np
import tensorflow as tf

from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper


class SoftTensorValueModel:

    dimensions = None
    string_type = None
    soft_by_dimension = None

    max_lengths = None
    soft_length_tensors = None

    tensor = None
    length_list = None

    def __init__(self, dimensions, string_type, max_lengths, soft_by_dimension):
        self.dimensions = dimensions
        self.string_type = string_type
        self.max_lengths = max_lengths
        self.soft_by_dimension = soft_by_dimension

    def get_data_type(self):
        return self.string_type

    def get_dimensions(self):
        return self.dimensions

    def get_dimension(self, idx):
        return self.dimensions[idx]

    def is_scalar(self):
        return len(self.dimensions) == 0

    def recursive_assign(self, python_representation, numpy_representation, current_prefix):
        local_python_representation = python_representation
        for idx in current_prefix:
            local_python_representation = local_python_representation[idx]

        if len(current_prefix) == len(self.max_lengths):
            numpy_representation[current_prefix] = local_python_representation
        else:
            for inner_elem_idx in range(len(local_python_representation)):
                self.recursive_assign(python_representation, numpy_representation, current_prefix + (inner_elem_idx, ))

    def recursive_length_list_retrieval(self, python_representation, numpy_representation, current_prefix):
        local_python_representation = python_representation
        for idx in current_prefix:
            local_python_representation = local_python_representation[idx]

        if len(current_prefix) == len(numpy_representation.shape):
            numpy_representation[current_prefix] = len(local_python_representation)
        else:
            for inner_elem_idx in range(len(local_python_representation)):
                self.recursive_length_list_retrieval(python_representation, numpy_representation, current_prefix + (inner_elem_idx, ))

    def recursive_max_length_retrieval(self, python_representation, needed_depth):
        if needed_depth == 0:
            return len(python_representation)
        else:
            inner_max_lengths = [self.recursive_max_length_retrieval(x, needed_depth-1) for x in python_representation]
            return max(inner_max_lengths)

    def initial_assign(self, python_representation):
        shape_initializer = [None] * len(self.max_lengths)
        for idx, mlen in enumerate(self.max_lengths):
            shape_initializer[idx] = self.recursive_max_length_retrieval(python_representation, needed_depth=idx)
            if mlen is not None:
                shape_initializer[idx] = min(mlen, shape_initializer[idx])

        numpy_representation = np.zeros(shape_initializer, dtype=self.get_numpy_type())
        if self.get_data_type() == "string":
            numpy_representation.fill("")

        self.recursive_assign(python_representation, numpy_representation, ())
        self.tensor = numpy_representation

        self.soft_length_tensors = [None] * len(self.max_lengths)
        self.max_lengths = shape_initializer

        for dim_idx, dim_is_soft in enumerate(self.soft_by_dimension):
            if dim_is_soft:
                dim_shape = shape_initializer[:dim_idx]
                numpy_representation = np.zeros(dim_shape, np.int32)
                self.recursive_length_list_retrieval(python_representation, numpy_representation, ())
                self.soft_length_tensors[dim_idx] = numpy_representation

    def assign(self, tensor, length_list, chop_dimensions=False):
        #Stupid sanity check:
        tensor.shape

        self.tensor = tensor

        if length_list is None:
            self.soft_length_tensors = [None] * len(self.get_max_lengths())
        else:
            self.soft_length_tensors = length_list

        if chop_dimensions:
            slc = [slice(None)] * len(self.tensor.shape)
            for idx, length_list in enumerate(self.soft_length_tensors):
                if length_list is not None:
                    max_length = length_list.max()
                    self.max_lengths[idx] = max_length

                    slc[idx] = slice(0, max_length)

            self.tensor = self.tensor[slc]


    def get_value(self):
        return self.tensor

    def get_lengths(self):
        return self.soft_length_tensors

    def get_max_lengths(self):
        return self.max_lengths

    def cast(self, new_type):
        if new_type == "float":
            new_value_model = SoftTensorValueModel(self.dimensions, "float", self.max_lengths, self.soft_by_dimension)
            new_value_model.assign(np.array(self.tensor).astype(np.float32), length_list=self.soft_length_tensors)
        if new_type == "int":
            new_value_model = SoftTensorValueModel(self.dimensions, "int", self.max_lengths, self.soft_by_dimension)
            new_value_model.assign(np.array(self.tensor).astype(np.int32), length_list=self.soft_length_tensors)

        return new_value_model

    def get_tensorflow_output_tensors(self):
        return [self.tensor, self.length_list]

    def apply_dropouts(self, dropout_rate, dropout_dim=None):
        keep_prob = 1 - float(dropout_rate)

        noise_shape = None
        if dropout_dim is not None:
            in_shape = tf.shape(self.tensor)
            kept_shape_dims = in_shape[:dropout_dim+1]
            dropped_out_dims = in_shape[dropout_dim+1:]

            noise_shape = tf.concat([kept_shape_dims, tf.ones_like(dropped_out_dims)], axis=-1)

        self.tensor = tf.nn.dropout(self.tensor, keep_prob=keep_prob, noise_shape=noise_shape)
        return self

    def get_tensorflow_type(self):
        tf_type = None
        if self.string_type == "int":
            tf_type = tf.int32
        elif self.string_type == "float":
            tf_type = tf.float32
        elif self.string_type == "string":
            tf_type = tf.string
        return tf_type

    def get_numpy_type(self):
        np_type = None
        if self.string_type == "int":
            np_type = np.int32
        elif self.string_type == "float":
            np_type = np.float32
        elif self.string_type == "string":
            np_type = np.object
        return np_type

    def format_for_program_output(self):
        sth = SoftTensorHelper()

        return sth.format_to_python_list(self.tensor, self.soft_length_tensors)