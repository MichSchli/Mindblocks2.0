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
        value = SequenceCrossEntropyValue()

        if "average_across_timesteps" in value_dictionary:
            value.set_average_across_timesteps(value_dictionary["average_across_timesteps"][0][0] == "True")

        if "average_across_batch" in value_dictionary:
            value.set_average_across_batch(value_dictionary["average_across_batch"][0][0] == "True")

        return value

    def execute(self, input_dictionary, value, output_value_models, mode):
        mask = tf.sequence_mask(input_dictionary["labels"].get_sequence_lengths(),
                                maxlen=input_dictionary["labels"].get_maximum_sequence_length(),
                                dtype=tf.float32)

        cross_entropy = tf.contrib.seq2seq.sequence_loss(
            logits=input_dictionary["logits"].get_value(),
            targets=input_dictionary["labels"].get_value(),
            average_across_timesteps=value.average_across_timesteps,
            average_across_batch=value.average_across_batch,
            weights=mask
        )

        if not value.average_across_batch or not value.average_across_timesteps:
            cross_entropy = tf.reduce_sum(cross_entropy)

        output_value_models["output"].assign(cross_entropy)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"output": TensorTypeModel("float", [])}

class SequenceCrossEntropyValue(ExecutionComponentValueModel):

    average_across_timesteps = None
    average_across_batch = None

    def __init__(self):
        self.average_across_timesteps = True
        self.average_across_batch = True

    def set_average_across_timesteps(self, b):
        self.average_across_timesteps = b

    def set_average_across_batch(self, b):
        self.average_across_batch = b