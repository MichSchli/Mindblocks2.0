from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class Accuracy(ComponentTypeModel):

    name = "Accuracy"
    in_sockets = ["predictions", "labels", "normalization"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return AccuracyValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        predictions = input_dictionary["predictions"].get_value()
        labels = input_dictionary["labels"].get_value().astype(np.int32).flatten()

        accuracy = (np.abs(predictions - labels) < value.tolerance).sum() / input_dictionary["normalization"].get_value()

        output_value_models["output"].assign(accuracy)

        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"output": TensorTypeModel("float", [None])}


class AccuracyValue(ExecutionComponentValueModel):

    tolerance = 1e-8