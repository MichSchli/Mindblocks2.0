from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Mean(ComponentTypeModel):

    name = "Mean"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return MeanValue()

    def execute(self, input_dictionary, value, mode):
        return {"output": tf.reduce_mean(input_dictionary["input"], axis=-1)}

    def build_value_type(self, input_types, value):
        return {"output": input_types["input"].copy()}

class MeanValue(ExecutionComponentValueModel):

    pass