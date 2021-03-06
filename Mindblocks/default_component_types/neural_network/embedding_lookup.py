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

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        vectors = input_dictionary["vectors"].get_value()

        idx = input_dictionary["indexes"].get_value()
        idx_len = input_dictionary["indexes"].get_lengths()

        vec_len = input_dictionary["vectors"].get_lengths()
        out_len = idx_len + vec_len[1:]

        lookup = tf.nn.embedding_lookup(vectors, idx)

        output_models["output"].assign(lookup, length_list=out_len)
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        vector_type = input_types["vectors"]
        index_type = input_types["indexes"]

        new_type = index_type.copy()
        new_type.set_data_type(vector_type.get_data_type())
        new_type.add_dimension(-1, vector_type.get_dimension(-1))

        return {"output": new_type}

class EmbeddingLookupValue(ExecutionComponentValueModel):

    pass