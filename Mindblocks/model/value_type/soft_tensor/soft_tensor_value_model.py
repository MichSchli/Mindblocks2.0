import numpy as np
import tensorflow as tf

from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper


class SoftTensorValueModel:
    """
    Class representing a "soft" tensor with specified lengths along one or several axes.
    """

    dimensions = None  # Original dimensions as given by type
    max_lengths = None  # Max lengths of current tensor

    string_type = None
    soft_by_dimension = None

    soft_length_tensors = None

    tensor = None

    language = None

    def __init__(self, dimensions, string_type, max_lengths, soft_by_dimension, language):
        self.dimensions = dimensions
        self.string_type = string_type
        self.max_lengths = max_lengths
        self.soft_by_dimension = soft_by_dimension
        self.language = language

    def get_language(self):
        return self.language

    def get_data_type(self):
        return self.string_type

    def get_dimensions(self):
        return self.dimensions

    def get_dimension(self, idx):
        return self.dimensions[idx]

    def is_scalar(self):
        return len(self.dimensions) == 0

    def initial_assign(self, python_representation):
        sth = SoftTensorHelper()
        tensor, soft_length_tensors = sth.to_soft_tensor(python_representation, self.max_lengths,
                                                         self.soft_by_dimension, self.get_data_type())

        self.assign(tensor, soft_length_tensors, chop_dimensions=True)

    def assign(self, tensor, length_list, chop_dimensions=False):
        # Sanity check:

        if self.get_language() == "python":
            n_tensor_dims = len(tensor.shape)
        else:
            n_tensor_dims = len(tensor.get_shape().as_list())
        n_required_dims = len(self.get_dimensions())

        if n_tensor_dims != n_required_dims:
            print("TENSOR DIMS OFF: ")
            print(n_tensor_dims)
            print(n_required_dims)
            exit()
        if length_list is not None:
            n_length_dims = len(length_list)
            if n_length_dims != n_required_dims:
                print("LENGTH DIMS OFF: ")
                print(n_length_dims)
                print(n_required_dims)
                exit()

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
                else:
                    self.max_lengths[idx] = self.tensor.shape[idx]

            self.tensor = self.tensor[tuple(slc)]

    def get_value(self):
        return self.tensor

    def get_lengths(self):
        if self.soft_length_tensors is None:
            return [None] * len(self.dimensions)

        return self.soft_length_tensors

    def get_max_lengths(self):
        return self.max_lengths

    def cast(self, new_type):
        sth = SoftTensorHelper()
        if new_type == "float":
            apply_fn = lambda x: float(x)
            result = sth.transform(self.tensor,
                                   self.soft_length_tensors,
                                   apply_fn,
                                   new_type=np.float32,
                                   transform_dim=-1)

        elif new_type == "int":
            apply_fn = lambda x: int(x)
            result = sth.transform(self.tensor,
                                   self.soft_length_tensors,
                                   apply_fn,
                                   new_type=np.int32,
                                   transform_dim=-1)

        elif new_type == "string":
            apply_fn = lambda x: str(x)
            result = sth.transform(self.tensor,
                                   self.soft_length_tensors,
                                   apply_fn,
                                   new_type=np.object,
                                   transform_dim=-1)
        elif new_type == "bool":
            apply_fn = lambda x: x == "True"
            result = sth.transform(self.tensor,
                                   self.soft_length_tensors,
                                   apply_fn,
                                   new_type=np.bool,
                                   transform_dim=-1)

        new_value_model = SoftTensorValueModel(self.dimensions, new_type, self.max_lengths, self.soft_by_dimension,
                                               language=self.language)
        new_value_model.assign(result, length_list=self.soft_length_tensors)

        return new_value_model

    def apply_dropouts(self, dropout_rate, dropout_dim=None):
        keep_prob = 1 - float(dropout_rate)

        noise_shape = None
        if dropout_dim is not None:
            in_shape = tf.shape(self.tensor)
            kept_shape_dims = in_shape[:dropout_dim + 1]
            dropped_out_dims = in_shape[dropout_dim + 1:]

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

    """
    Helpers:
    """

    def replace_elements_outside_lengths(self, outside_replacement):
        val = self.get_value()
        all_lengths = self.get_lengths()

        sth = SoftTensorHelper()
        replaced_val = sth.replace_elements_outside_lengths(val, all_lengths, outside_replacement)

        return replaced_val
