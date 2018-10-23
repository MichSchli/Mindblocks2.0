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
            if False:
                lengths = input_dictionary["input"].lengths if input_dictionary["input"].is_value_type("list") else input_dictionary["input"].get_sequence_lengths()
                max_length = tf.shape(val)[1]
                mask = tf.sequence_mask(lengths,
                                        maxlen=max_length,
                                        dtype=tf.bool)
                val = tf.where(mask, val, tf.ones_like(val)*tf.float32.min)

            argmax = tf.argmax(val, axis=-1, output_type=tf.int32)
        else:
            #print(input_dictionary["input"].get_value())
            argmax = np.argmax(input_dictionary["input"].get_value(), axis=-1)
            max = np.max(input_dictionary["input"].get_value(), axis=-1)

        print("Warning: Argmax breaks lengths")
        output_value_models["output"].assign(argmax, length_list=None)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()
        output_type.set_dimension(-1, 1, is_soft=False)
        return {"output": output_type}

class ArgmaxValue(ExecutionComponentValueModel):

    language = None

    def __init__(self, language):
        self.language = language