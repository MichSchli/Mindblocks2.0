from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel


class ExplicitIndexer(ComponentTypeModel):

    name = "ExplicitIndexer"
    in_socket_names = ["input"]
    out_socket_names = ["output"]

    def __init__(self):
        pass

    def get_new_value(self):
        return ExplicitIndexerValue()

    def execute(self, in_sockets, value, language="python"):
        return [[[value.do_index(v) for v in val] for val in in_sockets[0]]]

class ExplicitIndexerValue(ComponentValueModel):

    index = None

    def __init__(self):
        self.index = {}

    def load(self, value_lines):
        if "index" in value_lines:
            for string, _ in value_lines["index"]:
                self.index[string] = len(self.index)

    def do_index(self, value):
        return self.index[value]

    def copy(self):
        copy = ExplicitIndexerValue()
        copy.index = self.index
        return copy

    def describe(self):
        return "text=\""+self.text+"\""
