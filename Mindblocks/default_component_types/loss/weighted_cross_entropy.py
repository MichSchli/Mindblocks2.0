import tensorflow as tf

from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class WeightedCrossEntropy(ComponentTypeModel):

    name = "WeightedCrossEntropy"
    in_sockets = ["labels", "logits", "weights"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return WeightedCrossEntropyValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(input_dictionary["labels"].get_value(), tf.int32),
                logits=input_dictionary["logits"].get_value())

        lengths = input_dictionary["labels"].get_lengths()

        weights = input_dictionary["weights"].get_value()
        cross_entropy = self.scale_loss_by_weights(cross_entropy, weights)

        sth = SoftTensorHelper()
        replacement_tensor = tf.zeros_like(cross_entropy)
        cross_entropy = sth.replace_elements_outside_lengths(cross_entropy, lengths, replacement_tensor)

        for i in range(1, len(lengths)):
            if lengths[i] is not None:
                sum = tf.reduce_sum(cross_entropy, axis=i)

                axis_lengths = tf.cast(lengths[i], tf.float32)
                for _ in range(i + 1, len(lengths)):
                    axis_lengths = tf.expand_dims(axis_lengths, -1)
                cross_entropy = sum / (tf.cast(axis_lengths, tf.float32) + 1e-8)
            else:
                cross_entropy = tf.reduce_mean(cross_entropy, axis=i)

        output_value_models["output"].assign(cross_entropy, length_list=[lengths[0]])

        return output_value_models

    def scale_loss_by_weights(self, loss, weights):
        for dim in range(len(weights.shape), len(loss.shape)):
            weights = tf.expand_dims(weights, -1)
        loss = tf.multiply(loss, weights)
        return loss

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SoftTensorTypeModel([None], string_type="float")}

class WeightedCrossEntropyValue(ExecutionComponentValueModel):

    pass