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
        mean = tf.reduce_mean(input_dictionary["input"].get_value(), axis=value.axis)

        old_lengths = input_dictionary["input"].get_lengths()
        previous_dim_idxs = list(range(len(old_lengths)))
        del previous_dim_idxs[value.axis]

        new_lengths = [old_lengths[x] for x in previous_dim_idxs]

        output_value_models["output"].assign(mean, length_list=new_lengths)

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