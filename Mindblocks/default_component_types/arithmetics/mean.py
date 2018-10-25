from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Mean(ComponentTypeModel):

    name = "Mean"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return MeanValue(int(value_dictionary["axis"][0][0]))

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        all_lengths = input_dictionary["input"].get_lengths()
        axis_lengths = all_lengths[value.axis]

        val = input_dictionary["input"].get_value()
        replacement = tf.zeros_like(val)

        sth = SoftTensorHelper()
        zeroed_out = sth.replace_elements_outside_lengths(val, all_lengths, replacement)

        summed_dim = tf.reduce_sum(zeroed_out, axis=value.axis)
        mean_dim = summed_dim / axis_lengths

        old_lengths = input_dictionary["input"].get_lengths()
        previous_dim_idxs = list(range(len(old_lengths)))
        del previous_dim_idxs[value.axis]

        new_lengths = [old_lengths[x] for x in previous_dim_idxs]

        output_value_models["output"].assign(mean_dim, length_list=new_lengths)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        previous_dim_idxs = list(range(len(input_types["input"].get_dimensions())))
        del previous_dim_idxs[value.axis]

        output_type = input_types["input"].get_subtype(previous_dim_idxs)

        return {"output": output_type}

class MeanValue(ExecutionComponentValueModel):

    axis = None

    def __init__(self, axis):
        self.axis = axis