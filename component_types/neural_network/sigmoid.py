from model.component.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

class Sigmoid(ComponentTypeModel):

    name = "Sigmoid"
    in_socket_names = ["input"]
    out_socket_names = ["output"]
    available_languages = ["tensorflow"]

    def __init__(self):
        pass

    def execute(self, in_sockets, value, language="python"):
        return [tf.nn.sigmoid(in_sockets[0])]