from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel


class Constant(ComponentTypeModel):

    name = "Constant"
    in_socket_names = []
    out_socket_names = ["output"]

    def __init__(self):
        pass

    def get_new_value(self):
        return ConstantValue()

    def execute(self, value):
        print(value.text)


class ConstantValue(ComponentValueModel):

    value = None

    def __init__(self):
        self.value = 0.0

    def load(self, value_lines):
        if "text" in value_lines:
            self.text = value_lines["text"][0][0]

    def describe(self):
        return "value=\""+self.text+"\""
