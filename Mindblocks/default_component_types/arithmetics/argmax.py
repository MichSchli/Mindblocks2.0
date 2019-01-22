from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
import tensorflow as tf
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from operator import mul

from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class Argmax(ComponentTypeModel):

    name = "Argmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = ArgmaxValue(language)

        if "reduce_dimensions" in value_dictionary:
            str_dimensions = value_dictionary["reduce_dimensions"][0][0]
            int_dimensions = [int(s) for s in str_dimensions.split(",")]

            value.set_reduce_dimensions(int_dimensions)

        return value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        if value.language == "tensorflow":
            val = input_dictionary["input"].get_value()
            lengths = input_dictionary["input"].get_lengths()

            outside_replacement = tf.ones_like(val) * tf.float32.min
            val = input_dictionary["input"].replace_elements_outside_lengths(outside_replacement)

            val_shape = tf.shape(val)

            # Transpose the tensor to place all reduce dimensions at end:
            reduce_dimensions = value.reduce_dimensions
            all_dims = list(range(len(lengths)))
            true_reduce_dimensions = [all_dims[d] for d in reduce_dimensions]
            keep_dims = [d for d in all_dims if d not in true_reduce_dimensions]
            reshuffle_dims = keep_dims + true_reduce_dimensions
            reshuffled_val = tf.transpose(val, reshuffle_dims)

            # Flatten all reduce dimensions:
            argmax_shape = tf.stack([val_shape[d] for d in keep_dims] + [-1], -1)
            flattened_val = tf.reshape(reshuffled_val, argmax_shape)

            # Compute argmax:
            argmax = tf.argmax(flattened_val, axis=-1, output_type=tf.int32)

            # Initialize produce of all inner reduce dimensions:
            inner_dim_product = 1
            for d in true_reduce_dimensions[1:]:
                inner_dim_product *= val_shape[d]

            # Get all argmaxes by dims:
            argmaxes = []
            remainder = argmax
            for d in true_reduce_dimensions[1:]:
                argmax_this_dim = tf.floordiv(remainder, inner_dim_product)
                argmaxes.append(argmax_this_dim)
                remainder = remainder % inner_dim_product
                inner_dim_product /= val_shape[d]
            argmaxes.append(remainder)

            # Compute final output :
            final_lengths = [lengths[d] for d in keep_dims]
            if len(argmaxes) == 1:
                final_argmax = argmaxes[0]
            else:
                final_argmax = tf.stack(argmaxes, -1)
                final_lengths.append(None)

            output_value_models["output"].assign(final_argmax, length_list=final_lengths)

        else:
            print("Warning: Python argmax is broken. Use tensorflow or refactor.")
            argmax = np.argmax(input_dictionary["input"].get_value(), axis=-1)
            output_value_models["output"].assign(argmax, length_list=None)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()

        if len(value.reduce_dimensions) == 1:
            output_type.delete_dimension(-1)
            output_type.set_data_type("int")
        else:
            previous_dimensions = output_type.get_dimensions()
            reduce_dimensions = value.reduce_dimensions
            all_dims = list(range(len(previous_dimensions)))
            true_reduce_dimensions = [all_dims[d] for d in reduce_dimensions]
            keep_dims = [d for d in all_dims if d not in true_reduce_dimensions]

            old_sbd = output_type.get_soft_by_dimensions()

            new_dims = [previous_dimensions[d] for d in keep_dims] + [len(value.reduce_dimensions)]
            new_sbd = [old_sbd[d] for d in keep_dims] + [None]

            output_type = SoftTensorTypeModel(new_dims, soft_by_dimensions=new_sbd, string_type="int")

        output_type.set_name(value.get_name() + ":output")
        return {"output": output_type}

class ArgmaxValue(ExecutionComponentValueModel):

    language = None
    reduce_dimensions = None

    def __init__(self, language):
        self.language = language
        self.reduce_dimensions = [-1]

    def set_reduce_dimensions(self, dimensions):
        self.reduce_dimensions = dimensions