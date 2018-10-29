from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class HingeLoss(ComponentTypeModel):

    name = "HingeLoss"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return HingeLossValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        logits = input_dictionary["logits"].get_value()
        labels = tf.cast(tf.squeeze(input_dictionary["labels"].get_value()), tf.float32)
        loss = tf.losses.hinge_loss(logits=logits, labels=labels, reduction=tf.losses.Reduction.NONE)

        lengths = input_dictionary["labels"].get_lengths()

        sth = SoftTensorHelper()
        replacement_tensor = tf.zeros_like(loss)
        loss = sth.replace_elements_outside_lengths(loss, lengths[:-1], replacement_tensor)

        for i in range(1, len(lengths)):
            if lengths[i] is not None:
                sum = tf.reduce_sum(loss, axis=i)

                axis_lengths = tf.cast(lengths[i], tf.float32)
                for _ in range(i + 1, len(lengths)):
                    axis_lengths = tf.expand_dims(axis_lengths, -1)
                loss = sum / (tf.cast(axis_lengths, tf.float32) + 1e-8)
            else:
                loss = tf.reduce_mean(loss, axis=i)

        output_value_models["output"].assign(loss, length_list=[lengths[0]])

        if input_dictionary["labels"].is_value_type("list"):
            lengths = input_dictionary["labels"].lengths
            max_length = tf.reduce_max(lengths)
            mask = tf.sequence_mask(lengths,
                                    maxlen=max_length,
                                    dtype=tf.bool)
            loss = loss[:, :max_length]

            replacement = tf.zeros_like(loss)
            loss = tf.where(mask, loss, replacement)

            sum = tf.reduce_sum(loss, axis=-1)
            # TODO: Hack to deal with zero length sequences should be smarter:
            loss = sum / (tf.cast(lengths, tf.float32) + 1e-8)

        else:
            loss = tf.reduce_mean(loss, axis=-1)
        output_value_models["output"].assign(loss)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SoftTensorTypeModel([None], string_type="float")}

class HingeLossValue(ExecutionComponentValueModel):

    pass