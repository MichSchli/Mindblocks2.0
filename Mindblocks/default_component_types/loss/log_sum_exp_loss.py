import tensorflow as tf

from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class LogSumExpLoss(ComponentTypeModel):

    name = "LogSumExpLoss"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return LogSumExpLossValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        logits = input_dictionary["logits"].get_value()
        labels = tf.cast(input_dictionary["labels"].get_value(), tf.float32)

        # First, ensure labels and logits are zero outside label lengths:
        lengths = input_dictionary["labels"].get_lengths()
        sth = SoftTensorHelper()
        replacement_tensor = tf.zeros_like(labels)
        logits = sth.replace_elements_outside_lengths(logits, lengths, replacement_tensor)
        labels = sth.replace_elements_outside_lengths(labels, lengths, replacement_tensor)

        if value.aggregation == "flatten":
            logits = tf.reshape(logits, tf.stack([tf.shape(logits)[0], -1]))
            labels = tf.reshape(labels, tf.stack([tf.shape(labels)[0], -1]))

            VERY_NEGATIVE_NUMBER = -1e20
            log_scores_for_positive_labels = tf.reduce_logsumexp(logits + VERY_NEGATIVE_NUMBER * (1 - labels), axis=-1)
            log_norm = tf.reduce_logsumexp(logits, axis=-1)

            # output is negative log probability:
            scores = - ( log_scores_for_positive_labels - log_norm )

        output_value_models["output"].assign(scores, length_list=[lengths[0]])
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SoftTensorTypeModel([None], string_type="float")}

class LogSumExpLossValue(ExecutionComponentValueModel):

    aggregation = "flatten"