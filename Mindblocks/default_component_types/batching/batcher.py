import random

import numpy as np
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Batcher(ComponentTypeModel):

    name = "Batcher"
    in_sockets = ["data", "indexes"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return BatcherValue(value_dictionary["lazy"][0][0] == "True")

    def execute(self, input_dictionary, value, output_value_models, mode):
        data = np.array(input_dictionary["data"].get_value())
        indexes = input_dictionary["indexes"].get_value()

        output_value_models["output"].assign(data[indexes])
        return output_value_models

    def build_value_type_model(self, input_types, value):
        data_type = input_types["data"].copy()
        indexes_outer_dim = input_types["indexes"].get_dimensions()[0]
        data_type.subsample(indexes_outer_dim)
        return {"output": data_type}

    def has_batches(self, value, previous_values):
        return previous_values["indexes"]


class BatcherValue(ExecutionComponentValueModel):

    lazy = None

    def __init__(self, lazy):
        self.lazy = lazy