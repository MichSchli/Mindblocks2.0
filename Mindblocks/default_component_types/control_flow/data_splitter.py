import numpy as np
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel


class DataSplitter(ComponentTypeModel):

    name = "DataSplitter"
    in_sockets = ["input"]
    out_sockets = ["left", "right"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        return DataSplitterValue(int(value_dictionary["pivot"]))

    def execute(self, input_dictionary, value, mode):
        inp = input_dictionary["input"]
        inp = np.array(inp)
        return {"left": inp[:,:value.pivot+1], "right": inp[:,value.pivot+1:]}

    def infer_types(self, input_types, value):
        return {"left": input_types["input"], "right": input_types["input"]}

    def infer_dims(self, input_dims, value):
        return {"left": "abc", "right": "abc"}

class DataSplitterValue:

    pivot = None

    def __init__(self, pivot):
        self.pivot = pivot