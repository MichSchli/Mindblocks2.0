from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
import tensorflow as tf
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Argmax(ComponentTypeModel):

    name = "Argmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ArgmaxValue(language)

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        if value.language == "tensorflow":
            val = input_dictionary["input"].get_value()

            outside_replacement = tf.ones_like(val) * tf.float32.min
            val = input_dictionary["input"].replace_elements_outside_lengths(outside_replacement)

            argmax = tf.argmax(val, axis=-1, output_type=tf.int32)
        else:
            print("Warning: Python argmax breaks lengths")
            argmax = np.argmax(input_dictionary["input"].get_value(), axis=-1)

        output_value_models["output"].assign(argmax, length_list=None)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()
        output_type.delete_dimension(-1)
        output_type.set_name(value.get_name() + ":output")
        return {"output": output_type}

class ArgmaxValue(ExecutionComponentValueModel):

    language = None

    def __init__(self, language):
        self.language = language