import numpy as np
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class DataSplitter(ComponentTypeModel):

    name = "DataSplitter"
    in_sockets = ["input"]
    out_sockets = ["left", "right"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return DataSplitterValue(int(value_dictionary["pivot"][0][0]))

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        inp = input_dictionary["input"].get_value()
        inp = np.array(inp)

        output_value_models["left"].assign(inp[:,:value.pivot+1])
        output_value_models["right"].assign(inp[:,value.pivot+1:])

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left = input_types["input"].copy()
        right = input_types["input"].copy()
        left.set_dimension(-1, value.pivot+1)
        right.set_dimension(-1, right.dimensions[-1]-value.pivot-1)

        return {"left": left,
                "right": right}

class DataSplitterValue(ExecutionComponentValueModel):

    pivot = None

    def __init__(self, pivot):
        self.pivot = pivot