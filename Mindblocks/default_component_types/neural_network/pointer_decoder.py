from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class PointerDecoder(ComponentTypeModel):

    name = "PointerDecoder"
    in_sockets = ["word_logits", "word_indexes"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = PointerDecoderValue()
        value.language = language

        if "vocabulary_size" in value_dictionary:
            value.set_vocabulary_size(int(value_dictionary["vocabulary_size"][0][0]))

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        # Make sure logits outside lengths are 0:
        v = input_dictionary["word_logits"].get_value()
        l = input_dictionary["word_logits"].get_lengths()
        replacement = tf.zeros_like(v)
        sth = SoftTensorHelper()
        v = sth.replace_elements_outside_lengths(v, l, replacement)

        output_shape = tf.concat([tf.shape(v)[:-1], [value.get_vocabulary_size()]], axis=0)

        indexes = input_dictionary["word_indexes"].get_value()
        batch_range = tf.range(tf.shape(indexes)[0], dtype=tf.int32)
        batch_range = tf.tile(tf.expand_dims(batch_range, -1), [1, tf.shape(indexes)[1]])
        full_indexes = tf.stack([batch_range, indexes], axis=-1)

        output_tensor = tf.scatter_nd(full_indexes,
                                      v,
                                      output_shape)

        output_lengths = l[:]
        output_lengths[-1] = None

        output_tensor = tf.ones([tf.shape(v)[0],54]) + output_tensor

        output_models["output"].assign(output_tensor, output_lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        # logit input is batch x time. Output is batch x vocab size
        output = input_types["word_logits"].copy()
        output.set_dimension(-1, value.get_vocabulary_size(), is_soft=False)
        return {"output": output}


class PointerDecoderValue(ExecutionComponentValueModel):

    def set_vocabulary_size(self, size):
        self.vocabulary_size = size

    def get_vocabulary_size(self):
        return self.vocabulary_size