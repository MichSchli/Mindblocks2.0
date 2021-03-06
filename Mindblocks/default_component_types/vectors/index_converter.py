from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class IndexConverter(ComponentTypeModel):

    name = "IndexConverter"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = IndexConverterValue()

        if "keep_last_dimension" in value_dictionary:
            value.set_keep_last_dimension(value_dictionary["keep_last_dimension"][0][0] == "True")
        value.language = language

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.language == "python":
            print("didnt implement index_converter for python")
            exit()
        elif value.language == "tensorflow":
            v = input_dictionary["input"].get_value()
            lengths = input_dictionary["input"].get_lengths()[:]

            # Retrieve boolean length mask
            sth = SoftTensorHelper()
            if value.keep_last_dimension:
                length_mask = sth.retrieve_boolean_length_mask(tf.zeros(tf.shape(v)[:-1], dtype=tf.int32), lengths[:-1])
            else:
                length_mask = sth.retrieve_boolean_length_mask(v, lengths)

            # Use length mask to compute prefixes:
            prefixes = tf.cast(tf.where(length_mask), dtype=tf.int32)

            # Retrieve indexes and append to prefixes:
            final_indexes = tf.gather_nd(v, prefixes)
            if not value.keep_last_dimension:
                final_indexes = tf.expand_dims(final_indexes, -1)

            a_vecs = tf.unstack(prefixes, axis=-1)
            del a_vecs[-1]
            prefixes = tf.stack(a_vecs, -1)

            output = tf.concat([prefixes, final_indexes], axis=-1)

            output_models["output"].assign(output, length_list=[None, None])

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        in_type = input_types["input"].copy()
        n_dims = len(in_type.get_dimensions()) + 1
        out_type = SoftTensorTypeModel([None, n_dims], string_type="int")

        return {"output": out_type}


class IndexConverterValue(ExecutionComponentValueModel):
    def __init__(self):
        self.keep_last_dimension = False

    def set_keep_last_dimension(self, keep_last_dimension):
        self.keep_last_dimension = keep_last_dimension
