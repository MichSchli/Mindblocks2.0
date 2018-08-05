from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class EmbeddingLookup(ComponentTypeModel):

    name = "EmbeddingLookup"
    in_sockets = ["indexes", "vectors"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return EmbeddingLookupValue()

    def execute(self, input_dictionary, value, mode):
        return {"output": tf.nn.embedding_lookup(input_dictionary["vectors"],
                                                 input_dictionary["indexes"])}

    def build_value_type(self, input_types, value):
        vector_type = input_types["vectors"]
        index_type = input_types["indexes"]

        new_type = index_type.copy()
        new_type.set_data_type(vector_type.get_data_type())
        new_type.extend_dims(vector_type.get_inner_dim())
        return {"output": new_type}

    def infer_types(self, input_types, value):
        return {"output": input_types["vectors"]}

    def infer_dims(self, input_dims, value):
        return {"output": input_dims["vectors"]}

class EmbeddingLookupValue(ExecutionComponentValueModel):

    pass