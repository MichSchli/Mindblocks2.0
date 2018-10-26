from Mindblocks.error_handling.types.dimension_mismatch_exception import DimensionMismatchException
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
import tensorflow as tf
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class Concat(ComponentTypeModel):
    """
    Module representing concatenation of tensors along a given axis. Rules for dimensions:

    1) The number of dimensions of the left and right and right tensors must be the same.
    2) Every dimension except the concatenation axis must be equal, unknown, or 1.
    3) Dimensions of length 1 extend to cover their counterpart.
    4) Softness must match, except were dimension is 1.

    """

    name = "Concat"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ConcatValue(language)

    def handle_scalar(self, value, language):
        if language == "tensorflow":
            return tf.expand_dims(value, -1)
        else:
            return [value]

    def retrieve_expansion_dims(self, target_value, dimensions_to_expand, language):
        if language == "tensorflow":
            target_dims = tf.shape(target_value)
            baseline_dim = 1
        else:
            target_dims = target_value.shape
            baseline_dim = 1

        expansion_dims = [None] * len(dimensions_to_expand)
        for i, should_expand in enumerate(dimensions_to_expand):
            if should_expand:
                expansion_dims[i] = target_dims[i]
            else:
                expansion_dims[i] = baseline_dim

        return expansion_dims

    def expand_dims(self, value, target_dims, language):
        if language == "tensorflow":
            return tf.tile(value, target_dims)
        else:
            return np.tile(value, target_dims)

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        left_value = input_dictionary["left"].get_value()
        right_value = input_dictionary["right"].get_value()

        left_lengths = input_dictionary["left"].get_lengths()
        right_lengths = input_dictionary["right"].get_lengths()

        if input_dictionary["left"].is_scalar():
            left_value = self.handle_scalar(left_value, value.language)
            left_lengths = [None]

        if input_dictionary["right"].is_scalar():
            right_value = self.handle_scalar(right_value, value.language)
            right_lengths = [None]

        if value.should_expand("left"):
            dimensions_to_expand = value.get_dimensions_to_expand("left")
            expansion_dims = self.retrieve_expansion_dims(right_value, dimensions_to_expand, value.language)
            left_value = self.expand_dims(left_value, expansion_dims, value.language)

        if value.should_expand("right"):
            dimensions_to_expand = value.get_dimensions_to_expand("right")
            expansion_dims = self.retrieve_expansion_dims(left_value, dimensions_to_expand, value.language)
            right_value = self.expand_dims(right_value, expansion_dims, value.language)

        if value.language == "tensorflow":
            result = tf.concat([left_value, right_value], axis=-1, name=execution_component.get_name())
        else:
            result = np.concatenate((left_value, right_value))

        lengths = left_lengths
        for i in range(len(left_lengths)):
            if value.retrieve_lengths[i] == "right":
                lengths[i] = right_lengths[i]

        output_value_models["output"].assign(result, length_list=lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left_type = input_types["left"]
        right_type = input_types["right"]

        left_dimensions = left_type.get_dimensions()
        right_dimensions = right_type.get_dimensions()

        left_soft = left_type.get_soft_by_dimensions()
        right_soft = right_type.get_soft_by_dimensions()

        left_dim_string = left_type.get_dimension_string()
        right_dim_string = right_type.get_dimension_string()

        # Extend scalars to 1-dimensional tensors:
        left_dimensions = [1] if input_types["left"].is_scalar() else left_dimensions
        left_soft = [False] if input_types["left"].is_scalar() else left_soft
        right_dimensions = [1] if input_types["right"].is_scalar() else right_dimensions
        right_soft = [1] if input_types["right"].is_scalar() else right_soft

        # Verify the number of dimensions matches:
        if len(left_dimensions) != len(right_dimensions):
            raise DimensionMismatchException("Mismatched dimension shape in component "
                                             + value.get_name()+": "
                                             + left_dim_string
                                             + " does not match "
                                             + right_dim_string
                                             + ".")

        if value.axis < 0:
            value.axis += len(left_dimensions)

        left_is_hard = [(left_dimensions[idx] is not None and left_dimensions[idx] != 1) or left_soft[idx] for idx in range(len(left_dimensions))]
        right_is_hard = [(right_dimensions[idx] is not None and right_dimensions[idx] != 1) or right_soft[idx] for idx in range(len(left_dimensions))]

        # Verify there is no hard dimension mismatch:
        for i in range(len(left_dimensions)):
            if i == value.axis:
                continue
            if left_is_hard[i] and right_is_hard[i] and left_dimensions[i] != right_dimensions[i]:
                raise DimensionMismatchException("Mismatched hard dimensions along axis " + str(i) + " in component "
                                                 + value.get_name() + ": "
                                                 + left_dim_string
                                                 + " does not match "
                                                 + right_dim_string
                                                 + ".")

        # Determine where to retrieve potential length tensors:
        retrieve_lengths = [None] * len(left_dimensions)
        for i in range(len(left_dimensions)):
            if left_soft[i]:
                retrieve_lengths[i] = "left"
            elif right_soft[i]:
                retrieve_lengths[i] = "right"

        value.set_retrieve_lengths(retrieve_lengths)
        soft_by_dimension = [d is not None for d in retrieve_lengths]

        # Mark dimensions for expansion:
        expand_dims = [None] * len(left_dimensions)
        for i in range(len(left_dimensions)):
            if i == value.axis:
                pass
            elif left_is_hard[i] and right_dimensions[i] == 1:
                expand_dims[i] = "right"
            elif right_is_hard[i] and left_dimensions[i] == 1:
                expand_dims[i] = "left"

        value.set_expand_dims(expand_dims)

        # Determine final dimensionality:
        output_dimensions = [None] * len(left_dimensions)
        for i in range(len(left_dimensions)):
            if i == value.axis:
                if left_dimensions[i] is None or right_dimensions[i] is None:
                    output_dimensions[i] = None
                else:
                    output_dimensions[i] = left_dimensions[i] + right_dimensions[i]
            elif left_is_hard[i]:
                output_dimensions[i] = left_dimensions[i]
            elif right_is_hard[i]:
                output_dimensions[i] = right_dimensions[i]

        # Get string data type:
        data_type = input_types["left"].get_data_type()

        output_type = SoftTensorTypeModel(output_dimensions,
                                          soft_by_dimensions=soft_by_dimension,
                                          string_type=data_type)

        return {"output": output_type}


class ConcatValue(ExecutionComponentValueModel):

    axis = -1
    retrieve_lengths = None
    expand_dims = None

    def __init__(self, language):
        self.language = language

    def set_retrieve_lengths(self, retrieve_lengths):
        self.retrieve_lengths = retrieve_lengths

    def set_expand_dims(self, expand_dims):
        self.expand_dims = expand_dims

    def should_expand(self, side):
        for expand_dim in self.expand_dims:
            if expand_dim == side:
                return True
        return False

    def get_dimensions_to_expand(self, side):
        dims = [False] * len(self.expand_dims)
        for i, expand_dim in enumerate(self.expand_dims):
            if expand_dim == side:
                dims[i] = True

        return dims