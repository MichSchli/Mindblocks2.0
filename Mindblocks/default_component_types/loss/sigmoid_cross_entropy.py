from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class SigmoidCrossEntropy(ComponentTypeModel):

    name = "SigmoidCrossEntropy"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return SigmoidCrossEntropyValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        logits = input_dictionary["logits"].get_value()
        labels = tf.cast(tf.squeeze(input_dictionary["labels"].get_value()), tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)

        if input_dictionary["labels"].is_value_type("list"):
            lengths = input_dictionary["labels"].lengths
            max_length = tf.reduce_max(lengths)
            mask = tf.sequence_mask(lengths,
                                    maxlen=max_length,
                                    dtype=tf.float32)
            cross_entropy = cross_entropy[:, :max_length]
            cross_entropy *= mask

            sum = tf.reduce_sum(cross_entropy, axis=-1)
            # TODO: Hack to deal with zero length sequences should be smarter:
            cross_entropy = sum / (tf.cast(lengths, tf.float32) + 1e-8)
        else:
            cross_entropy = tf.reduce_mean(cross_entropy, axis=-1)
        output_value_models["output"].assign(cross_entropy)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": TensorTypeModel("float", [])}

class SigmoidCrossEntropyValue(ExecutionComponentValueModel):

    pass