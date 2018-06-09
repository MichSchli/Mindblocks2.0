from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel


class DebugPrint(ComponentTypeModel):

    name = "DebugPrint"
    in_socket_names = ["input"]
    out_socket_names = []

    def __init__(self):
        pass

    def get_new_value(self):
        return DebugPrintValue()

    def execute(self, in_sockets, value):
        debug_text = value.text

        if in_sockets[0] is not None:
            debug_text = debug_text.replace("$input", str(in_sockets[0].get_value()))

        print(debug_text)

        return []

class DebugPrintValue(ComponentValueModel):

    text = None

    def __init__(self):
        self.text = ""

    def load(self, value_lines):
        if "text" in value_lines:
            self.text = value_lines["text"][0][0]

    def copy(self):
        copy = DebugPrintValue()
        copy.text = self.text
        return copy

    def describe(self):
        return "text=\""+self.text+"\""
