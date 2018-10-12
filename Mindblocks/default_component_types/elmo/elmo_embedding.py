from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import tensorflow_hub as tf_hub

class ElmoEmbedding(ComponentTypeModel):

    name = "ElmoEmbedding"
    in_sockets = ["input"]
    out_sockets = ["word_embeddings", "sentence_embedding"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ElmoEmbeddingValue()

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        elmo = tf_hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        lengths = input_dictionary["input"].get_sequence_lengths()
        inputs = {"tokens": input_dictionary["input"].get_sequences(),
                  "sequence_len": lengths}

        embeddings = elmo(inputs=inputs, signature="tokens", as_dict=True)

        output_models["word_embeddings"].assign_with_lengths(embeddings["elmo"], lengths, language="tensorflow")
        output_models["sentence_embedding"].assign(embeddings["default"], language="tensorflow")

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        dim = 1024
        input_seq = input_types["input"]

        output_seq = SequenceBatchTypeModel("float", [dim], input_seq.get_batch_size(), None)
        output_sent = TensorTypeModel("float", [input_seq.get_batch_size(), dim])

        return {"word_embeddings": output_seq, "sentence_embedding": output_sent}


class ElmoEmbeddingValue(ExecutionComponentValueModel):

    def count_parameters(self):
        return 4