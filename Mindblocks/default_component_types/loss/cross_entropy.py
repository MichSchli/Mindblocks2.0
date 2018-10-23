from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class CrossEntropy(ComponentTypeModel):

    name = "CrossEntropy"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return CrossEntropyValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(tf.squeeze(input_dictionary["labels"].get_value()), tf.int32),
                logits=input_dictionary["logits"].get_value())

        lengths = input_dictionary["labels"].get_lengths()
        output_value_models["output"].assign(cross_entropy, lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SoftTensorTypeModel([None], string_type="float")}

class CrossEntropyValue(ExecutionComponentValueModel):

    pass