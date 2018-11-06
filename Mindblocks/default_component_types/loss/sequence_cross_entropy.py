import tensorflow as tf

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class SequenceCrossEntropy(ComponentTypeModel):

    name = "SequenceCrossEntropy"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = SequenceCrossEntropyValue()

        if "average_across_timesteps" in value_dictionary:
            value.set_average_across_timesteps(value_dictionary["average_across_timesteps"][0][0] == "True")

        return value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        mask = tf.sequence_mask(input_dictionary["labels"].get_lengths()[1],
                                maxlen=input_dictionary["labels"].get_max_lengths()[1],
                                dtype=tf.float32)

        cross_entropy = tf.contrib.seq2seq.sequence_loss(
            logits=input_dictionary["logits"].get_value(),
            targets=input_dictionary["labels"].get_value(),
            average_across_timesteps=value.average_across_timesteps,
            average_across_batch=False,
            weights=mask
        )

        if not value.average_across_timesteps:
            cross_entropy = tf.reduce_sum(cross_entropy, axis=-1)

        output_value_models["output"].assign(cross_entropy, length_list=None)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SoftTensorTypeModel([None], string_type="float")}

    def make_inferences(self, execution_value, tf_run_variables):
        pass


class SequenceCrossEntropyValue(ExecutionComponentValueModel):

    average_across_timesteps = None

    def __init__(self):
        self.average_across_timesteps = True

    def set_average_across_timesteps(self, b):
        self.average_across_timesteps = b