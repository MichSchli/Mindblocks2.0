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
        mask = tf.sequence_mask(input_dictionary["labels"].get_sequence_lengths(),
                                maxlen=input_dictionary["labels"].get_maximum_sequence_length(),
                                dtype=tf.float32)

        mask2 = tf.sequence_mask(input_dictionary["labels"].get_sequence_lengths(),
                                maxlen=input_dictionary["labels"].get_maximum_sequence_length(),
                                dtype=tf.int32)

        sm = tf.nn.softmax(input_dictionary["logits"].get_sequences())[:,:tf.shape(mask2)[-1]]
        ps = tf.cast(tf.argmax(sm, axis=-1), dtype=tf.int32)
        mask2 = tf.Print(mask2, [tf.shape(mask2), tf.shape(input_dictionary["labels"].get_sequences()), tf.shape(input_dictionary["logits"].get_sequences()), tf.shape(ps), tf.shape(sm)], message="shapes", summarize=100)

        preds = mask2 * ps
        labels = input_dictionary["labels"].get_sequences() * mask2
        mask = tf.Print(mask, [preds], message="preds", summarize=100)
        mask = tf.Print(mask, [labels], message="labels", summarize=100)
        diff = tf.reduce_sum(tf.where(tf.equal(preds, labels), tf.zeros_like(mask), tf.ones_like(mask)))
        mask = tf.Print(mask, [diff], message="missed", summarize=100)

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