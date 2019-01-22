import numpy as np
import tensorflow as tf

from Mindblocks.error_handling.types.dimension_mismatch_exception import DimensionMismatchException


class SoftTensorBinaryOperatorHelper:

    def get_combine_type_dimensions(self, left_type, right_type, op_name):
        """
        Method for combining the dimensions of the two arguments in a binary operation. Proceeds according to the following rules:

        1) If the number of dimensions is mismatched, throw an error.
        2) For each dimension, if both sides have dimensionality/max dimensionality > 1, throw an error if the dimensionality is mismatched.
        3) For each dimension, if both sides have dimensionality/max dimensionality > 1, throw an error if the softness is mismatched.
        4) For each dimension, if dim(X) = 1 and dim(Y) > 1, expand from Y.
        5) For each dimension, if dim(X) = 1 and Y is soft, receive softness from Y.

        :param left_type: The lefthand type of the binary operator
        :param right_type: The righthand type of the binary operator
        :param op_name: The name of the underlying operator for use in error reporting
        :return: A tuple containing (dimensions,softness) for the output type of the operator.
        """

        left_dimensions = left_type.get_dimensions()
        right_dimensions = right_type.get_dimensions()
        left_soft = left_type.get_soft_by_dimensions()
        right_soft = right_type.get_soft_by_dimensions()
        left_dim_string = left_type.get_dimension_string()
        right_dim_string = right_type.get_dimension_string()

        # Extend scalars to 1-dimensional tensors:
        left_dimensions = [1] if left_type.is_scalar() else left_dimensions
        left_soft = [False] if left_type.is_scalar() else left_soft
        right_dimensions = [1] if right_type.is_scalar() else right_dimensions
        right_soft = [False] if right_type.is_scalar() else right_soft

        # Verify the number of dimensions matches:
        if len(left_dimensions) != len(right_dimensions):
            raise DimensionMismatchException("Mismatched dimension shape in component "
                                             + op_name + ": "
                                             + left_dim_string
                                             + " does not match "
                                             + right_dim_string
                                             + ".")

        # Verify there is no dimension mismatch:
        for i in range(len(left_dimensions)):
            if left_dimensions[i] is None or left_dimensions[i] == 1:
                continue
            elif right_dimensions[i] is None or right_dimensions[i] == 1:
                continue
            elif left_dimensions[i] != right_dimensions[i]:
                raise DimensionMismatchException("Mismatched hard dimensions along axis " + str(i) + " in component "
                                                 + op_name + ": "
                                                 + left_dim_string
                                                 + " does not match "
                                                 + right_dim_string
                                                 + ".")

        # Verify there is no softness mismatch
        left_is_hard = [(left_dimensions[idx] is not None and left_dimensions[idx] != 1) or left_soft[idx] for idx in
                        range(len(left_dimensions))]
        right_is_hard = [(right_dimensions[idx] is not None and right_dimensions[idx] != 1) or right_soft[idx] for idx
                         in range(len(left_dimensions))]
        for i in range(len(left_dimensions)):
            if left_is_hard[i] != right_is_hard[i]:
                raise DimensionMismatchException("Mismatched softness along axis " + str(i) + " in component "
                                                 + op_name + ": "
                                                 + left_dim_string
                                                 + " does not match "
                                                 + right_dim_string
                                                 + ".")
        # Mark dimensions for expansion:
        expand_dims = [None] * len(left_dimensions)
        for i in range(len(left_dimensions)):
            if left_dimensions[i] > 0 and right_dimensions[i] == 1:
                expand_dims[i] = "right"
            elif right_dimensions[i] > 0 and left_dimensions[i] == 1:
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
            if expand_dims[i] is None or expand_dims[i] == "left":
                output_dimensions[i] = left_dimensions[i]
            else:
                output_dimensions[i] = right_dimensions[i]

        return output_dimensions, soft_by_dimension