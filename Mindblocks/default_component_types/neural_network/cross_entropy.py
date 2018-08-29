from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class CrossEntropy(ComponentTypeModel):

    name = "CrossEntropy"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return CrossEntropyValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(input_dictionary["labels"].get_value()),
                logits=input_dictionary["logits"].get_value())
        output_value_models["output"].assign(cross_entropy)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"output": TensorTypeModel("float", [])}

class CrossEntropyValue(ExecutionComponentValueModel):

    pass