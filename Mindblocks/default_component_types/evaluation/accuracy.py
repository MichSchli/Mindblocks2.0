import numpy as np

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class Accuracy(ComponentTypeModel):

    name = "Accuracy"
    in_sockets = ["predictions", "labels", "normalization"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return AccuracyValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        predictions = input_dictionary["predictions"].get_value()
        labels = input_dictionary["labels"].get_value().astype(np.int32).flatten()
        lengths = input_dictionary["labels"].get_lengths()

        accuracy = (np.abs(predictions - labels) < value.tolerance)

        output_value_models["output"].assign(accuracy, length_list=lengths)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SoftTensorTypeModel([None], string_type="float")}


class AccuracyValue(ExecutionComponentValueModel):

    tolerance = 1e-8