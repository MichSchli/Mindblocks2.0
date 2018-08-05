import numpy as np
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class DataSplitter(ComponentTypeModel):

    name = "DataSplitter"
    in_sockets = ["input"]
    out_sockets = ["left", "right"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        return DataSplitterValue(int(value_dictionary["pivot"][0]))

    def execute(self, input_dictionary, value, mode):
        inp = input_dictionary["input"]
        inp = np.array(inp)
        return {"left": inp[:,:value.pivot+1], "right": inp[:,value.pivot+1:]}

    def build_value_type(self, input_types, value):
        return {"left": input_types["input"].copy().set_inner_dim(None),
                "right": input_types["input"].copy().set_inner_dim(None)}

class DataSplitterValue(ExecutionComponentValueModel):

    pivot = None

    def __init__(self, pivot):
        self.pivot = pivot