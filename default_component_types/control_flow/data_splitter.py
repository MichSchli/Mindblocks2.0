import numpy as np
from model.component_type.component_type_model import ComponentTypeModel


class DataSplitter(ComponentTypeModel):

    name = "DataSplitter"
    in_sockets = ["input"]
    out_sockets = ["left", "right"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        return DataSplitterValue(int(value_dictionary["pivot"]))

    def execute(self, input_dictionary, value):
        inp = input_dictionary["input"]
        inp = np.array(inp)
        return {"left": inp[:,:value.pivot+1], "right": inp[:,value.pivot+1:]}

    def infer_types(self, input_types, value):
        return {"output": "abc"}

    def infer_dims(self, input_dims, value):
        return {"output": "abc"}

class DataSplitterValue:

    pivot = None

    def __init__(self, pivot):
        self.pivot = pivot