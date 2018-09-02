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

    def execute(self, input_dictionary, value, output_value_models, mode):
        if value.language == "tensorflow":
            left_value = input_dictionary["left"].get_value()
            right_value = input_dictionary["right"].get_value()

            for _ in range(value.should_expand_left):
                left_value = tf.expand_dims(left_value, -1)

            for _ in range(value.should_expand_right):
                right_value = tf.expand_dims(right_value, -1)

            if value.cover_list == "right":
                expand_to_cover = tf.shape(left_value)[1]
                right_value = tf.expand_dims(right_value, 1)
                nd = tf.concat([[1, expand_to_cover], tf.ones_like(tf.shape(right_value)[2:])], axis=-1)
                right_value = tf.tile(right_value, nd)

            result = tf.concat([left_value, right_value], axis=-1)
        elif value.new_array:
            result = np.array([input_dictionary["left"].get_value(), input_dictionary["right"].get_value()])
        else:
            result = np.concatenate((input_dictionary["left"].get_value(), input_dictionary["right"].get_value()))

        output_value_models["output"].assign(result, value.language)
        if value.out_type == "list":
            output_value_models["output"].lengths = input_dictionary["left"].get_lengths()
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left_type = input_types["left"]
        right_type = input_types["right"]

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

        output = input_types["left"].copy()

        if len(left_dims) > 0 and len(right_dims) > 0:
            output.set_inner_dim(left_dims[-1] + right_dims[-1])

        return {"output": output}


class ConcatValue(ExecutionComponentValueModel):

    new_array = False
    cover_list = None
    should_expand_left = 0
    should_expand_right = 0

    out_type = None

    def __init__(self, language):
        self.language = language

    def set_cover_list(self, inp):
        self.cover_list = inp