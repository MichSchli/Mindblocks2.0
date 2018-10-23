from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
import tensorflow as tf
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Concat(ComponentTypeModel):

    name = "Concat"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ConcatValue(language)

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        left_value = input_dictionary["left"].get_value()
        right_value = input_dictionary["right"].get_value()

        if value.language == "tensorflow":
            if input_dictionary["left"].is_scalar():
                left_value = tf.expand_dims(left_value, -1)
            if input_dictionary["right"].is_scalar():
                right_value = tf.expand_dims(right_value, -1)

            result = tf.concat([left_value, right_value], axis=-1, name=execution_component.get_name())
        else:
            if input_dictionary["left"].is_scalar():
                left_value = [left_value]
            if input_dictionary["right"].is_scalar():
                right_value = [right_value]

            result = np.concatenate((left_value, right_value))

        lengths = input_dictionary["left"].get_lengths()

        if len(lengths) == 0:
            lengths = [None]
        else:
            lengths[-1] = None

        output_value_models["output"].assign(result, length_list=lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left_type = input_types["left"]
        right_type = input_types["right"]

        """
        if left_type.is_value_type("list") and \
                right_type.is_value_type("list"):
            left_dims = input_types["left"].get_inner_dim()
            right_dims = input_types["right"].get_inner_dim()

            left_num_dims = len(input_types["left"].inner_shape)
            right_num_dims = len(input_types["right"].inner_shape)

            if left_num_dims > right_num_dims:
                value.should_expand_right = left_num_dims - right_num_dims
            elif left_num_dims < right_num_dims:
                value.should_expand_left = right_num_dims - left_num_dims

            value.out_type = "list"

            output = input_types["left"].copy()
            output.set_inner_dim(left_dims + right_dims if left_dims is not None and right_dims is not None else None)
            return {"output": output}
        if left_type.is_value_type("list") and \
                right_type.is_value_type("tensor"):
            left_dims = input_types["left"].get_inner_dim()
            right_dims = input_types["right"].get_inner_dim()

            value.set_cover_list("right")
            value.out_type = "list"

            output = input_types["left"].copy()
            output.set_inner_dim(left_dims + right_dims if left_dims is not None and right_dims is not None else None)
            return {"output": output}

        left_dims = input_types["left"].get_dimensions()
        right_dims = input_types["right"].get_dimensions()

        if len(left_dims) == 0 and len(right_dims) == 0:
            value.new_array = True
        """

        output = input_types["left"].copy()

        left_dim = 1 if input_types["left"].is_scalar() else  input_types["left"].get_dimension(-1)
        right_dim = 1 if input_types["right"].is_scalar() else  input_types["right"].get_dimension(-1)

        output.set_dimension(-1, left_dim + right_dim)

        return {"output": output}


class ConcatValue(ExecutionComponentValueModel):

    def __init__(self, language):
        self.language = language
