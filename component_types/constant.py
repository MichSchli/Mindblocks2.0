from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel


class Constant(ComponentTypeModel):

    name = "Constant"
    in_socket_names = []
    out_socket_names = ["output"]
    available_languages = ["python", "tensorflow"]

    def __init__(self):
        pass

    def get_new_value(self):
        return ConstantValue()

    def execute(self, in_sockets, value):
        return [value.value]


class ConstantValue(ComponentValueModel):

    value = None

    def __init__(self):
        self.value = 0.0

    def load(self, value_lines):
        if "value" in value_lines:
            self.value = float(value_lines["value"][0][0])

    def copy(self):
        copy = ConstantValue()
        copy.value = self.value
        return copy

    def describe(self):
        return "value=\""+self.value+"\""
