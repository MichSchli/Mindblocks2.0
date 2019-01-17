import sys
import tensorflow as tf
import tensorflow_hub as tf_hub

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel

import os

class ElmoEmbedding(ComponentTypeModel):

    name = "ElmoEmbedding"
    in_sockets = ["input"]
    out_sockets = ["word_embeddings", "sentence_embedding"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        if "elmo_dir" in value_dictionary:
            elmo_dir = value_dictionary["elmo_dir"][0][0]
        else:
            elmo_dir = None

        return ElmoEmbeddingValue(elmo_dir)

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.elmo_dir is not None:
            os.environ['TFHUB_CACHE_DIR'] = value.elmo_dir
            print("loading elmo to location: " + os.environ['TFHUB_CACHE_DIR'], file=sys.stderr)
        else:
            print("Warning: elmo dir not specified", file=sys.stderr)

        elmo = tf_hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        all_lengths = input_dictionary["input"].get_lengths()
        lengths = all_lengths[-1]

        tokens = input_dictionary["input"].get_value()

        if len(all_lengths) > 2:
            tokens_shape = tf.shape(tokens)
            tokens = tf.reshape(tokens, [-1, tokens_shape[-1]])
            lengths = tf.reshape(lengths, [-1])

        inputs = {"tokens": tokens,
                  "sequence_len": lengths}

        embeddings = elmo(inputs=inputs, signature="tokens", as_dict=True)

        word_embeddings = embeddings["elmo"]
        sentence_embedding = embeddings["default"]

        if len(all_lengths) > 2:
            word_embedding_shape = tf.concat([tokens_shape, [1024]], axis=0)
            sentence_embedding_shape = tf.concat([tokens_shape[:-1], [1024]], axis=0)

            word_embeddings = tf.reshape(word_embeddings, word_embedding_shape)
            sentence_embedding = tf.reshape(sentence_embedding, sentence_embedding_shape)

        output_models["word_embeddings"].assign(word_embeddings, length_list=all_lengths + [None])
        output_models["sentence_embedding"].assign(sentence_embedding, length_list=None)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        dim = 1024
        input_seq = input_types["input"]

        input_word_dimensions = input_seq.get_dimensions()[:]
        input_sentence_dimensions = input_word_dimensions[:-1]

        input_word_soft_by_dimension = input_seq.get_soft_by_dimensions()[:]
        input_sentence_soft_by_dimension = input_word_soft_by_dimension[:-1]

        output_seq = SoftTensorTypeModel(input_word_dimensions + [dim],
                                         string_type="float",
                                         soft_by_dimensions=input_word_soft_by_dimension + [False])

        output_sent = SoftTensorTypeModel(input_sentence_dimensions + [dim],
                                         string_type="float",
                                         soft_by_dimensions=input_sentence_soft_by_dimension + [False])

        return {"word_embeddings": output_seq, "sentence_embedding": output_sent}


class ElmoEmbeddingValue(ExecutionComponentValueModel):

    def __init__(self, elmo_dir):
        self.elmo_dir = elmo_dir

    def count_parameters(self):
        return 4