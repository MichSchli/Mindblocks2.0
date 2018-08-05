from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor_type import TensorType


class Accuracy(ComponentTypeModel):

    name = "Accuracy"
    in_sockets = ["predictions", "labels", "normalization"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return AccuracyValue()

    def execute(self, input_dictionary, value, mode):
        predictions = input_dictionary["predictions"]
        labels = input_dictionary["labels"].astype(np.int32).flatten()

        accuracy = (np.abs(predictions - labels) < value.tolerance).sum() / input_dictionary["normalization"]

        return {"output": accuracy}

    def build_value_type(self, input_types, value):
        return {"output": TensorType("float", [None])}


class AccuracyValue(ExecutionComponentValueModel):

    tolerance = 1e-8