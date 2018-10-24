from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow_hub as tf_hub

from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class ElmoEmbedding(ComponentTypeModel):

    name = "ElmoEmbedding"
    in_sockets = ["input"]
    out_sockets = ["word_embeddings", "sentence_embedding"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ElmoEmbeddingValue()

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        elmo = tf_hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        all_lengths = input_dictionary["input"].get_lengths()
        lengths = all_lengths[1]
        inputs = {"tokens": input_dictionary["input"].get_value(),
                  "sequence_len": lengths}

        embeddings = elmo(inputs=inputs, signature="tokens", as_dict=True)

        output_models["word_embeddings"].assign(embeddings["elmo"], length_list=all_lengths)
        output_models["sentence_embedding"].assign(embeddings["default"], length_list=None)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        dim = 1024
        input_seq = input_types["input"]

        output_seq = SoftTensorTypeModel([input_seq.get_dimension(0), None, dim],
                                         string_type="float",
                                         soft_by_dimensions=[False, True, False])
        output_sent = SoftTensorTypeModel([input_seq.get_dimension(0), dim], string_type="float")

        return {"word_embeddings": output_seq, "sentence_embedding": output_sent}


class ElmoEmbeddingValue(ExecutionComponentValueModel):

    def count_parameters(self):
        return 4