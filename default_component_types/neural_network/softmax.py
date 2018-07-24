from model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf


class Softmax(ComponentTypeModel):

    name = "Softmax"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return SoftmaxValue()

    def execute(self, input_dictionary, value):
        return {"output": tf.nn.softmax(input_dictionary["input"])}

    def infer_types(self, input_types, value):
        return {"output": input_types["input"]}

    def infer_dims(self, input_dims, value):
        return {"output": input_dims["input"]}

class SoftmaxValue:

    pass