from model.component.component_type.component_type_model import ComponentTypeModel
from model.graph.value_type_model import ValueTypeModel
import tensorflow as tf


class Add(ComponentTypeModel):

    name = "CrossEntropy"
    in_socket_names = ["logits", "labels"]
    out_socket_names = ["output"]
    available_languages = ["tensorflow"]

    def __init__(self):
        pass

    def execute(self, in_sockets, value, language="python"):
        return [tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=in_sockets[0], labels=in_sockets[1], dim=-1))]

    def evaluate_value_type(self, in_types, value):
        return [ValueTypeModel(in_types[0].type, in_types[0].dim[:-1])]