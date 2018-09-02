from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
import tensorflow as tf
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class Argmax(ComponentTypeModel):

    name = "Argmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ArgmaxValue(language)

    def execute(self, input_dictionary, value, output_value_models, mode):
        if value.language == "tensorflow":
            val = input_dictionary["input"].get_value()
            if input_dictionary["input"].is_value_type("list") or input_dictionary["input"].is_value_type("sequence"):
                lengths = input_dictionary["input"].lengths if input_dictionary["input"].is_value_type("list") else input_dictionary["input"].get_sequence_lengths()
                max_length = tf.shape(val)[1]
                mask = tf.sequence_mask(lengths,
                                        maxlen=max_length,
                                        dtype=tf.bool)
                val = tf.where(mask, val, tf.ones_like(val)*tf.float32.min)

            print(val)
            val = tf.Print(val, [tf.shape(val)], message="vals", summarize=100)
            argmax = tf.argmax(val, axis=-1, output_type=tf.int32)
            print(argmax)
            argmax = tf.Print(argmax, [tf.shape(argmax), argmax], message="argmax "+value.get_name(), summarize=100)
        else:
            argmax = np.argmax(input_dictionary["input"].get_value(), axis=-1)
        output_value_models["output"].assign(argmax, value.language)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        if input_types["input"].is_value_type("list"):
            output_type = TensorTypeModel(input_types["input"].data_type,
                                          [None,None])
        elif input_types["input"].is_value_type("sequence"):
            output_type = TensorTypeModel(input_types["input"].type,
                                          [input_types["input"].get_batch_size()])
            print(output_type.dimensions)
        else:
            output_type = input_types["input"].copy()
            output_type.set_inner_dim(1)
        return {"output": output_type}

class ArgmaxValue(ExecutionComponentValueModel):

    language = None

    def __init__(self, language):
        self.language = language