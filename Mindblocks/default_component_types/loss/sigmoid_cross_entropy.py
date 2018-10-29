from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class SigmoidCrossEntropy(ComponentTypeModel):

    name = "SigmoidCrossEntropy"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return SigmoidCrossEntropyValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        logits = input_dictionary["logits"].get_value()
        labels = tf.cast(input_dictionary["labels"].get_value(), tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)

        lengths = input_dictionary["labels"].get_lengths()

        sth = SoftTensorHelper()
        replacement_tensor = tf.zeros_like(cross_entropy)
        cross_entropy = sth.replace_elements_outside_lengths(cross_entropy, lengths[:-1], replacement_tensor)

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

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SoftTensorTypeModel([None], string_type="float")}

class SigmoidCrossEntropyValue(ExecutionComponentValueModel):

    pass