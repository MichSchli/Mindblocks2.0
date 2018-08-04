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

    def infer_types(self, input_types, value):
        return {"output": input_types["vectors"]}

    def infer_dims(self, input_dims, value):
        return {"output": input_dims["vectors"]}

class EmbeddingLookupValue(ExecutionComponentValueModel):

    pass