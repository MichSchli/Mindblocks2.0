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

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        data = np.array(input_dictionary["data"].get_value())
        lengths = input_dictionary["data"].get_lengths()
        indexes = input_dictionary["indexes"].get_value()

        batch_lengths = [None]*len(lengths)
        for i in range(len(lengths)):
            if lengths[i] is not None:
                batch_lengths[i] = lengths[i][indexes]

        batch = data[indexes]
        self.log("Batch: " + str(batch), "batching", "batches")

        output_value_models["output"].assign(batch, length_list=batch_lengths, chop_dimensions=True)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        data_type = input_types["data"].copy()

        data_type.set_dimension(0, None)

        for idx, is_soft in enumerate(data_type.get_soft_by_dimensions()):
            if is_soft:
                data_type.set_dimension

        return {"output": data_type}

    def has_batches(self, value, previous_values, mode):
        return previous_values["indexes"]


class BatcherValue(ExecutionComponentValueModel):

    lazy = None

    def __init__(self, lazy):
        self.lazy = lazy