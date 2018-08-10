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

    def execute(self, input_dictionary, value, output_value_models, mode):
        mean = tf.reduce_mean(input_dictionary["input"].get_value(), axis=value.axis)
        output_value_models["output"].assign(mean, language="tensorflow")

        return output_value_models

    def build_value_type_model(self, input_types, value):
        if input_types["input"].is_value_type("sequence") and value.axis == 1:
            output_type = input_types["input"].get_single_token_type()
            output_type.extend_outer_dim(input_types["input"].get_batch_size())
        else:
            output_type = input_types["input"].copy()
            output_type.set_inner_dim(1)
        return {"output": output_type}

class MeanValue(ExecutionComponentValueModel):

    axis = None

    def __init__(self, axis):
        self.axis = axis