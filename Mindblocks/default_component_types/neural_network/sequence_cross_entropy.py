from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class SequenceCrossEntropy(ComponentTypeModel):

    name = "SequenceCrossEntropy"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return SequenceCrossEntropyValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        mask = tf.sequence_mask(input_dictionary["labels"].get_sequence_lengths(), dtype=tf.float32)
        cross_entropy = tf.contrib.seq2seq.sequence_loss(
                logits=input_dictionary["logits"].get_value(),
                targets=input_dictionary["labels"].get_value(),
                weights=mask
        )
        output_value_models["output"].assign(cross_entropy)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"output": TensorTypeModel("float", [])}

class SequenceCrossEntropyValue(ExecutionComponentValueModel):

    pass