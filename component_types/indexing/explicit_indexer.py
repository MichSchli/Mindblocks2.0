from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel
from model.graph.value_type_model import ValueTypeModel
import numpy as np

class ExplicitIndexer(ComponentTypeModel):

    name = "ExplicitIndexer"
    in_socket_names = ["input"]
    out_socket_names = ["output"]

    def __init__(self):
        pass

    def get_new_value(self):
        return ExplicitIndexerValue()

    def execute(self, in_sockets, value, language="python"):
        return [np.squeeze([[value.do_index(v) for v in val] for val in in_sockets[0]])]

    def evaluate_value_type(self, in_types, value):
        return [ValueTypeModel("int" if not value.use_onehot else "float", [None, None])]

class ExplicitIndexerValue(ComponentValueModel):

    index = None
    use_onehot = None

    def __init__(self):
        self.index = {}
        self.use_onehot = False

    def load(self, value_lines):
        if "index" in value_lines:
            for string, _ in value_lines["index"]:
                self.index[string] = len(self.index)

        if "onehot_encoding" in value_lines:
            self.use_onehot = value_lines["onehot_encoding"][0][0] == "True"

    def do_index(self, value):
        if not self.use_onehot:
            return self.index[value]
        else:
            onehot_length = len(self.index)
            onehot = [0] * onehot_length
            onehot[self.index[value]] = 1
            return onehot


    def copy(self):
        copy = ExplicitIndexerValue()
        copy.index = self.index
        copy.use_onehot = self.use_onehot
        return copy

    def describe(self):
        return "text=\""+self.text+"\""
