import random

import numpy as np

from Mindblocks.helpers.logging.logger_factory import LoggerFactory
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

        batch = data[indexes]
        self.log("Batch: " + str(batch), "batching", "batches")

        output_value_models["output"].assign(batch)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        data_type = input_types["data"].copy()
        data_type.subsample(None)
        return {"output": data_type}

    def has_batches(self, value, previous_values):
        return previous_values["indexes"]


class BatcherValue(ExecutionComponentValueModel):

    lazy = None

    def __init__(self, lazy):
        self.lazy = lazy