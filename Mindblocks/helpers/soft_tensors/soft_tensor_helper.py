import numpy as np
import tensorflow as tf

class SoftTensorHelper:

    def recursive_transform(self, input_tensor, output_tensor, current_prefix, length_tensor_list, stop_dim, transform_fn):
        """
        Recursively traverses the tensor to apply the transformation.

        :param input_tensor:
        :param output_tensor:
        :param current_prefix:
        :param length_tensor_list:
        :param stop_dim:
        :param transform_fn:
        :return:
        """
        local_old_representation = self.extract_local_subtensor(input_tensor, current_prefix)

        if len(current_prefix) == stop_dim or len(current_prefix) - len(input_tensor.shape) -1 == stop_dim:
            output_tensor[tuple(current_prefix)] = transform_fn(local_old_representation)
        else:
            local_length = length_tensor_list[len(current_prefix)]
            if local_length is not None:
                for idx in current_prefix:
                    local_length = local_length[idx]
            else:
                local_length = local_old_representation.shape[0]

            for inner_elem_idx in range(local_length):
                self.recursive_transform(input_tensor, output_tensor, current_prefix + (inner_elem_idx,), length_tensor_list, stop_dim, transform_fn)

    def transform(self, input_tensor, length_tensor_list, transform, new_type=np.float32, transform_dim=-1):
        """
        Applies a transformation in the form of a lambda term to every element of a soft tensor.

        :param input_tensor: The full input tensor in the form of a numpy array.
        :param length_tensor_list: The list of length tensors associated with the input tensor.
        :param transform: The lambda transform to apply.
        :param new_type: The numpy type of the elements after transformation. Defaults to float.
        :param transform_dim: The dimension along which to apply the transformation.
        :return: A transformed tensor with the same soft dimensions as the input tensor.
        """

        if transform_dim == -1:
            new_dims = input_tensor.shape
        else:
            new_dims = input_tensor.shape[:transform_dim + 1]
        new_tensor = np.zeros(new_dims, dtype=new_type)

        self.recursive_transform(input_tensor, new_tensor, (), length_tensor_list, transform_dim, transform)

        return new_tensor

    def recursive_transform_combine(self, first_tensor, second_tensor, output_tensor, current_prefix, length_tensor_list, stop_dim, transform_fn):
        """
        Recursively traverses the tensor to apply the transformation.

        :param first_tensor:
        :param second_tensor:
        :param output_tensor:
        :param current_prefix:
        :param length_tensor_list:
        :param stop_dim:
        :param transform_fn:
        :return:
        """
        local_first_representation = self.extract_local_subtensor(first_tensor, current_prefix)
        local_second_representation = self.extract_local_subtensor(second_tensor, current_prefix)

        if len(current_prefix) == stop_dim or len(current_prefix) - len(first_tensor.shape) -1 == stop_dim:
            output_tensor[tuple(current_prefix)] = transform_fn(local_first_representation, local_second_representation)
        else:
            local_length = length_tensor_list[len(current_prefix)]
            if local_length is not None:
                for idx in current_prefix:
                    local_length = local_length[idx]
            else:
                local_length = local_first_representation.shape[0]

            for inner_elem_idx in range(local_length):
                self.recursive_transform_combine(first_tensor, second_tensor, output_tensor, current_prefix + (inner_elem_idx,), length_tensor_list, stop_dim, transform_fn)

    def transform_combine(self, first_tensor, second_tensor, length_tensor_list, transform, new_type=np.float32, transform_dim=-1):
        """
        Applies a transformation in the form of a lambda term to every pair of elements from two soft tensors.

        :param first_tensor: The full first input tensor in the form of a numpy array.
        :param second_tensor: The full second input tensor in the form of a numpy array.
        :param length_tensor_list: The list of length tensors associated with the input tensor.
        :param transform: The lambda transform to apply.
        :param new_type: The numpy type of the elements after transformation. Defaults to float.
        :param transform_dim: The dimension along which to apply the transformation.
        :return: A transformed tensor with the same soft dimensions as the input tensor.
        """

        if transform_dim == -1:
            new_dims = first_tensor.shape
        else:
            new_dims = first_tensor.shape[:transform_dim + 1]
        new_tensor = np.zeros(new_dims, dtype=new_type)

        self.recursive_transform_combine(first_tensor, second_tensor, new_tensor, (), length_tensor_list, transform_dim, transform)

        return new_tensor

    def recursive_python_list_build(self, input_tensor, length_tensor_list, recursion_prefix):
        """
        Recursively builds a python list-of-lists representation for a soft tensor.

        :param input_tensor: The full input tensor in the form of a numpy array.
        :param length_tensor_list: The list of length tensors associated with the input tensor.
        :param recursion_prefix: A prefix tuple representing the coordinates in the input tensor being worked on.
        :return: A python list-of-list format version of the part of the input tensor specified by the prefix.
        """

        local_input_subtensor = self.extract_local_subtensor(input_tensor, recursion_prefix)

        old_rep_shape = np.array(input_tensor).shape
        if len(recursion_prefix) == len(old_rep_shape):
            return local_input_subtensor
        else:
            local_length = length_tensor_list[len(recursion_prefix)]
            if local_length is not None:
                for idx in recursion_prefix:
                    local_length = local_length[idx]
            else:
                local_length = np.array(local_input_subtensor).shape[0]

            curr_list = []
            for inner_elem_idx in range(local_length):
                curr_list.append(self.recursive_python_list_build(input_tensor, length_tensor_list, recursion_prefix + (inner_elem_idx,)))

            return curr_list

    def extract_local_subtensor(self, input_tensor, prefix_coordinates):
        """
        Extract a local subtensor according to a specified prefix

        :param input_tensor: The full input tensor of shape (d1, ..., dn) being worked on.
        :param prefix_coordinates: Prefix for retrieving the tensor in the form an m-dimensional tuple.
        :return: The subtensor of dimension (dm, ..., dn) located a the specified coordinates
        """
        local_input_subtensor = input_tensor
        for idx in prefix_coordinates:
            local_input_subtensor = local_input_subtensor[idx]
        return local_input_subtensor

    def format_to_python_list(self, input_tensor, length_tensor_list):
        """
        Formats a soft tensor to a python list. Used for program output.

        :param input_tensor: The input tensor in the form of a numpy array.
        :param length_tensor_list: The list of length tensors associated with the input tensor.
        :return: A python list-of-list format version of the input tensor.
        """

        rec_build = self.recursive_python_list_build(input_tensor, length_tensor_list, ())

        return rec_build

    def retrieve_boolean_length_mask(self, input_tensor, length_tensor_list):
        """
        Retrieves a mask tensor containing "true" inside the specified lengths and "false" outside.

        :param input_tensor: The tensor to be masked in the form of a tensorflow tensor
        :param length_tensor_list: The list of length tensors associated with the input tensor
        :return: A mask of dtype bool.
        """

        true_tensor = tf.ones_like(input_tensor, dtype=tf.bool)
        false_tensor = tf.zeros_like(input_tensor, dtype=tf.bool)

        return self.replace_elements_outside_lengths(true_tensor, length_tensor_list, false_tensor)

    def replace_elements_outside_lengths(self, input_tensor, length_tensor_list, replacement_tensor):
        """
        Fill the margins of a soft tensor with a specified value (requires tensorflow):

        :param input_tensor: The tensor prior to filling in the form of a tensorflow tensor.
        :param length_tensor_list: The list of length tensors associated with the input tensor.
        :param replacement_tensor: A pre-specified tensor used for replacement.
        :return: A modified version of the input tensor with the margins tiled according to the replacement value.
        """

        for idx, length in enumerate(length_tensor_list):
            if length is not None:
                max_length = tf.shape(input_tensor)[idx]
                mask = tf.sequence_mask(length,
                                        maxlen=max_length,
                                        dtype=tf.bool)

                for dim in range(idx + 1, len(length_tensor_list)):
                    mask = tf.expand_dims(mask, -1)

                target_dims = tf.ones(idx+1, dtype=tf.int32)
                target_dims = tf.concat((target_dims, tf.shape(input_tensor)[idx+1:]), axis=-1)

                mask = tf.tile(mask, target_dims)

                input_tensor = tf.where(mask, input_tensor, replacement_tensor)
        return input_tensor

    def python_replace_elements_outside_lengths(self, input_tensor, length_tensor_list, replacement_tensor):
        """
        Fill the margins of a soft tensor with a specified value (requires python):

        :param input_tensor: The tensor prior to filling in the form of a tensorflow tensor.
        :param length_tensor_list: The list of length tensors associated with the input tensor.
        :param replacement_tensor: A pre-specified tensor used for replacement.
        :return: A modified version of the input tensor with the margins tiled according to the replacement value.
        """

        r = replacement_tensor.copy()
        self.recursive_transform(input_tensor, r, (), length_tensor_list, -1, lambda x: x)

        return r

    def recursive_assign(self, python_representation, numpy_representation, current_prefix):
        local_python_representation = python_representation
        for idx in current_prefix:
            local_python_representation = local_python_representation[idx]

        if len(current_prefix) == len(numpy_representation.shape) - 1:
            local_length = len(local_python_representation)
            numpy_representation[tuple(current_prefix)][:local_length] = local_python_representation
        else:
            for inner_elem_idx in range(len(local_python_representation)):
                self.recursive_assign(python_representation, numpy_representation, current_prefix + (inner_elem_idx, ))

    def recursive_length_list_retrieval(self, python_representation, numpy_representation, current_prefix):
        local_python_representation = python_representation
        for idx in current_prefix:
            local_python_representation = local_python_representation[idx]

        if len(current_prefix) == len(numpy_representation.shape):
            numpy_representation[tuple(current_prefix)] = len(local_python_representation)
        else:
            for inner_elem_idx in range(len(local_python_representation)):
                self.recursive_length_list_retrieval(python_representation, numpy_representation, current_prefix + (inner_elem_idx, ))

    def recursive_max_length_retrieval(self, python_representation, needed_depth):
        if needed_depth == 0:
            return len(python_representation)
        else:
            inner_max_lengths = [self.recursive_max_length_retrieval(x, needed_depth-1) for x in python_representation]
            return max(inner_max_lengths) if len(inner_max_lengths) > 0 else 0

    def to_soft_tensor(self, python_representation, max_lengths, soft_by_dimension, data_type):
        if len(max_lengths) < 2:
            tensor = np.array(python_representation, dtype=self.get_numpy_type(data_type))
            soft_length_tensors = [None] * len(max_lengths)
            return tensor, soft_length_tensors

        shape_initializer = [None] * len(max_lengths)
        for idx, mlen in enumerate(max_lengths):
            shape_initializer[idx] = self.recursive_max_length_retrieval(python_representation, needed_depth=idx)
            if mlen is not None:
                shape_initializer[idx] = min(mlen, shape_initializer[idx])

        numpy_representation = np.zeros(shape_initializer, dtype=self.get_numpy_type(data_type))
        if data_type == "string":
            numpy_representation.fill("")

        self.recursive_assign(python_representation, numpy_representation, ())
        tensor = numpy_representation

        soft_length_tensors = [None] * len(max_lengths)

        for dim_idx, dim_is_soft in enumerate(soft_by_dimension):
            if dim_is_soft:
                dim_shape = shape_initializer[:dim_idx]
                numpy_representation = np.zeros(dim_shape, np.int32)
                self.recursive_length_list_retrieval(python_representation, numpy_representation, ())
                soft_length_tensors[dim_idx] = numpy_representation

        return tensor, soft_length_tensors

    def get_numpy_type(self, string_type):
        np_type = None
        if string_type == "int":
            np_type = np.int32
        elif string_type == "float":
            np_type = np.float32
        elif string_type == "string":
            np_type = np.object
        return np_type