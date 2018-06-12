from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel
import tensorflow as tf

from model.graph.value_type_model import ValueTypeModel


class Constant(ComponentTypeModel):

    name = "Constant"
    in_socket_names = []
    out_socket_names = ["output"]
    available_languages = ["python", "tensorflow"]

    def __init__(self):
        pass

    def get_new_value(self):
        return ConstantValue()

    def execute(self, in_sockets, value, language="python"):
        if language == "tensorflow":
            return [tf.constant(value.value)]
        else:
            return [value.value]

    def evaluate_value_type(self, in_types, value):
        return [ValueTypeModel(value.type, 1)]


class ConstantValue(ComponentValueModel):

    value = None
    type = None

    def __init__(self):
        self.value = 0.0
        self.type = "float"

    def load(self, value_lines):
        if "value" in value_lines:
            self.value = float(value_lines["value"][0][0])
        if "type" in value_lines:
            self.value = value_lines["type"][0][0]

    def copy(self):
        copy = ConstantValue()
        copy.value = self.value
        copy.type = self.type
        return copy

    def describe(self):
        return "value=\""+self.value+"\""
