import random

import numpy as np
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Batcher(ComponentTypeModel):

    name = "Batcher"
    in_sockets = ["data", "indexes"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return BatcherValue(value_dictionary["lazy"][0] == "True")

    def execute(self, input_dictionary, value, mode):
        data = np.array(input_dictionary["data"])
        indexes = input_dictionary["indexes"]

        if value.lazy:
            return {"output": data[indexes]}

        return None

    def build_value_type(self, input_types, value):
        data_type = input_types["data"].copy()
        indexes_outer_dim = input_types["indexes"].get_outer_dim_size()
        data_type.set_outer_dim_size(indexes_outer_dim)
        return {"output": data_type}

    def infer_types(self, input_types, value):
        return {"output": "string"}

    def infer_dims(self, input_dims, value):
        return {"output": [None, None]}


class BatcherValue(ExecutionComponentValueModel):

    lazy = None

    def __init__(self, lazy):
        self.lazy = lazy