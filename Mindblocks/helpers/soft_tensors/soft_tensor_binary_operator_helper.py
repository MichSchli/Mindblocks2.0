import numpy as np
import tensorflow as tf

from Mindblocks.error_handling.types.dimension_mismatch_exception import DimensionMismatchException
from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class SoftTensorBinaryOperatorHelper:

    def create_output_type(self, left_type, right_type, output_string_type, op_name):
        output_dimensions, output_softness = self.get_combine_type_dimensions(left_type, right_type, op_name)
        return SoftTensorTypeModel(output_dimensions,
                                   string_type=output_string_type,
                                   soft_by_dimensions=output_softness,
                                   reference_name=op_name + "_output")

    def get_combine_type_dimensions(self, left_type, right_type, op_name):
        """
        Method for combining the dimensions of the two arguments in a binary operation. Proceeds according to the following rules:

        1) If the number of dimensions is mismatched, throw an error.
        2) For each dimension, if both sides have dimensionality/max dimensionality > 1, throw an error if the dimensionality is mismatched.
        3) For each dimension, if both sides have dimensionality/max dimensionality > 1, throw an error if the softness is mismatched.
        4) For each dimension, if dim(X) = 1 and dim(Y) > 1, expand from Y.
        5) For each dimension, if dim(X) = 1 and Y is soft, receive softness from Y.
        6) Unknown dimensions (e.g. None) are filled if possible, but will expand to singular dimensions.

        :param left_type: The lefthand type of the binary operator
        :param right_type: The righthand type of the binary operator
        :param op_name: The name of the underlying operator for use in error reporting
        :return: A tuple containing (dimensions,softness) for the output type of the operator.
        """

        if left_type.is_scalar() and right_type.is_scalar():
            return [], []

        left_dim_string = left_type.get_dimension_string()
        right_dim_string = right_type.get_dimension_string()

        left_dimensions, left_soft, right_dimensions, right_soft = self.extract_dimension_info(left_type, right_type)

        # Handle partial dimensions:
        if len(left_dimensions) != len(right_dimensions):
            output_dimensions, soft_by_dimension = self.handle_partial_dimension_match(left_dim_string, left_type,
                                                                                       op_name, right_dim_string,
                                                                                       right_type)
        else:
            output_dimensions, soft_by_dimension = self.combine_dimensions(left_type, right_type, left_dim_string,
                                                                       right_dim_string, op_name)

        return output_dimensions, soft_by_dimension

    def handle_partial_dimension_match(self, left_dim_string, left_type, op_name, right_dim_string, right_type):
        left_dimensions, left_soft, right_dimensions, right_soft = self.extract_dimension_info(left_type,
                                                                                               right_type)
        if len(left_dimensions) > len(right_dimensions):
            partial_dims = list(range(len(right_dimensions)))
            left_subtype = left_type.get_subtype(partial_dims)

            partial_output_dims, partial_output_softness = self.combine_dimensions(left_subtype, right_type,
                                                                                   left_dim_string,
                                                                                   right_dim_string,
                                                                                   op_name)

            remaining_output_dims = left_dimensions[len(right_dimensions):]
            remaining_output_softness = left_soft[len(right_dimensions):]

            output_dimensions = partial_output_dims + remaining_output_dims
            soft_by_dimension = partial_output_softness + remaining_output_softness
        elif len(left_dimensions) < len(right_dimensions):
            partial_dims = list(range(len(left_dimensions)))
            right_subtype = right_type.get_subtype(partial_dims)

            partial_output_dims, partial_output_softness = self.combine_dimensions(right_subtype, left_type,
                                                                                   left_dim_string,
                                                                                   right_dim_string,
                                                                                   op_name)

            remaining_output_dims = right_dimensions[len(left_dimensions):]
            remaining_output_softness = right_soft[len(left_dimensions):]

            output_dimensions = partial_output_dims + remaining_output_dims
            soft_by_dimension = partial_output_softness + remaining_output_softness
        return output_dimensions, soft_by_dimension

    def extract_dimension_info(self, left_type, right_type):
        left_dimensions = left_type.get_dimensions()
        right_dimensions = right_type.get_dimensions()
        left_soft = left_type.get_soft_by_dimensions()
        right_soft = right_type.get_soft_by_dimensions()

        # Extend scalars to 1-dimensional tensors:
        left_dimensions = [1] if left_type.is_scalar() else left_dimensions
        left_soft = [False] if left_type.is_scalar() else left_soft
        right_dimensions = [1] if right_type.is_scalar() else right_dimensions
        right_soft = [False] if right_type.is_scalar() else right_soft

        return left_dimensions, left_soft, right_dimensions, right_soft

    def combine_dimensions(self, left_type, right_type, left_dim_string, right_dim_string, op_name):
        left_dimensions, left_soft, right_dimensions, right_soft = self.extract_dimension_info(left_type, right_type)

        # Verify there is no dimension mismatch:
        for i in range(len(left_dimensions)):
            if left_soft[i] or left_dimensions[i] is None or left_dimensions[i] == 1:
                continue
            elif right_soft[i] or right_dimensions[i] is None or right_dimensions[i] == 1:
                continue
            elif left_dimensions[i] != right_dimensions[i]:
                raise DimensionMismatchException("Mismatched hard dimensions along axis " + str(i) + " in component "
                                                 + op_name + ": "
                                                 + left_dim_string
                                                 + " does not match "
                                                 + right_dim_string
                                                 + ".")

        # Verify there is no soft dimension mismatch:
        for i in range(len(left_dimensions)):
            if left_dimensions[i] is None or right_dimensions[i] is None:
                continue
            if left_soft[i] and right_soft[i] and left_dimensions[i] != right_dimensions[i]:
                raise DimensionMismatchException("Mismatched soft dimensions along axis " + str(i) + " in component "
                                                 + op_name + ": "
                                                 + left_dim_string
                                                 + " does not match "
                                                 + right_dim_string
                                                 + ".")

        # Verify there is no softness mismatch
        left_must_be_hard = [not left_soft[idx] and not left_dimensions[idx] == 1 for idx in
                             range(len(left_dimensions))]
        right_must_be_hard = [not right_soft[idx] and not right_dimensions[idx] == 1 for idx in
                              range(len(left_dimensions))]
        for i in range(len(left_dimensions)):
            if (left_must_be_hard[i] and right_soft[i]) or (right_must_be_hard[i] and left_soft[i]):
                raise DimensionMismatchException("Mismatched softness along axis " + str(i) + " in component "
                                                 + op_name + ": "
                                                 + left_dim_string
                                                 + " does not match "
                                                 + right_dim_string
                                                 + ".")
        # Mark dimensions for expansion:
        expand_dims = [None] * len(left_dimensions)
        for i in range(len(left_dimensions)):
            if (left_dimensions[i] is None or left_dimensions[i]) > 0 and right_dimensions[i] == 1:
                expand_dims[i] = "right"
            elif (right_dimensions[i] is None or right_dimensions[i] > 0) and left_dimensions[i] == 1:
                expand_dims[i] = "left"

        # Determine where to retrieve potential length tensors:
        retrieve_lengths = [None] * len(left_dimensions)
        for i in range(len(left_dimensions)):
            if left_soft[i]:
                retrieve_lengths[i] = "left"
            elif right_soft[i]:
                retrieve_lengths[i] = "right"
        soft_by_dimension = [d is not None for d in retrieve_lengths]

        # Determine final dimensionality:
        output_dimensions = [None] * len(left_dimensions)
        for i in range(len(left_dimensions)):
            if expand_dims[i] is None:
                output_dimensions[i] = left_dimensions[i] if left_dimensions[i] is not None else right_dimensions[i]
            elif expand_dims[i] == "right":
                output_dimensions[i] = left_dimensions[i]
            else:
                output_dimensions[i] = right_dimensions[i]
        return output_dimensions, soft_by_dimension

    def process(self, left_value_model, right_value_model, op, output_value_model, language="python"):
        left_lengths = left_value_model.get_lengths()
        right_lengths = right_value_model.get_lengths()

        new_lengths = [None] * max(len(left_lengths), len(right_lengths))
        length_origins = [None for _ in new_lengths]

        for i in range(len(left_lengths)):
            if new_lengths[i] is None and left_lengths[i] is not None:
                new_lengths[i] = left_lengths[i]
                length_origins[i] = "left"

        for i in range(len(right_lengths)):
            if new_lengths[i] is None and right_lengths[i] is not None:
                new_lengths[i] = right_lengths[i]
                length_origins[i] = "right"

        left_value = left_value_model.get_value()
        right_value = right_value_model.get_value()

        if language == "tensorflow":
            left_dims = len(left_value.shape)
            right_dims = len(right_value.shape)

            all_left_dims = left_value_model.get_dimensions()[:]
            all_right_dims = right_value_model.get_dimensions()[:]

            for dim in range(left_dims, right_dims):
                left = tf.expand_dims(left, -1)
                all_left_dims.append(1)

            for dim in range(right_dims, left_dims):
                right = tf.expand_dims(right, -1)
                all_right_dims.append(1)

            for i in range(max(left_dims, right_dims)):
                if all_left_dims[i] == 1 and all_right_dims[i] != all_left_dims[i]:
                    dims_to_add = tf.shape(right_value)[i]
                    expand_origin = "right"
                elif all_right_dims[i] == 1 and all_right_dims[i] != all_left_dims[i]:
                    dims_to_add = tf.shape(left_value)[i]
                    expand_origin = "left"
                else:
                    continue

                for l in range(i, max(left_dims, right_dims)):
                    if new_lengths[l] is not None and length_origins[l] != expand_origin:
                        length_expansion = [1] * l
                        length_expansion[i] *= dims_to_add
                        new_lengths[l] = tf.tile(new_lengths[l], length_expansion)

        new_value = op(left_value, right_value)

        if language=="tensorflow":
            replacement = tf.zeros_like(new_value)
            sth = SoftTensorHelper()
            new_value = sth.replace_elements_outside_lengths(new_value, new_lengths, replacement)

        output_value_model.assign(new_value, length_list=new_lengths)

        return output_value_model
