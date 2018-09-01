from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class EmbeddingLookup(ComponentTypeModel):

    name = "EmbeddingLookup"
    in_sockets = ["indexes", "vectors"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return EmbeddingLookupValue()

    def execute(self, input_dictionary, value, output_models, mode):
        if input_dictionary["indexes"].is_value_type("tensor"):
            lookup = tf.nn.embedding_lookup(input_dictionary["vectors"].get_value(),
                                        input_dictionary["indexes"].get_value())

            output_models["output"].assign(lookup)
        elif input_dictionary["indexes"].is_value_type("sequence"):
            idx = input_dictionary["indexes"].get_sequence()
            lookup = tf.nn.embedding_lookup(input_dictionary["vectors"].get_value(),
                                            idx)

            lengths = input_dictionary["indexes"].get_sequence_lengths()

            output_models["output"].assign_with_lengths(lookup, lengths)
        elif input_dictionary["indexes"].is_value_type("list"):
            idx = input_dictionary["indexes"].get_items()
            lookup = tf.nn.embedding_lookup(input_dictionary["vectors"].get_value(),
                                            idx)

            lengths = input_dictionary["indexes"].get_lengths()

            output_models["output"].assign_with_lengths(lookup, lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        vector_type = input_types["vectors"]
        index_type = input_types["indexes"]

        new_type = index_type.copy()
        new_type.set_data_type(vector_type.get_data_type())
        new_type.extend_dims(vector_type.get_inner_dim())

        return {"output": new_type}

class EmbeddingLookupValue(ExecutionComponentValueModel):

    pass