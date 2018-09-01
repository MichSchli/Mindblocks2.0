import numpy as np
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class DataSlicer(ComponentTypeModel):

    name = "DataSlicer"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return DataSlicerValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        exit()
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        input_copy = input_types["input"].copy()
        input_copy.set_inner_dim(1)
        return {"output": input_copy}

class DataSlicerValue(ExecutionComponentValueModel):

    pass