from model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf


class CrossEntropy(ComponentTypeModel):

    name = "CrossEntropy"
    in_sockets = ["labels", "logits"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return CrossEntropyValue()

    def execute(self, input_dictionary, value, mode):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(input_dictionary["labels"]), logits=input_dictionary["logits"]))
        return {"output": cross_entropy}

    def infer_types(self, input_types, value):
        return {"output": "float"}

    def infer_dims(self, input_dims, value):
        return {"output": 1}

class CrossEntropyValue:

    pass