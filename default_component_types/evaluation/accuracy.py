from model.component_type.component_type_model import ComponentTypeModel
import numpy as np


class Accuracy(ComponentTypeModel):

    name = "Accuracy"
    in_sockets = ["predictions", "labels"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return AccuracyValue()

    def execute(self, input_dictionary, value):
        predictions = input_dictionary["predictions"]
        labels = input_dictionary["labels"].astype(np.int32).flatten()

        accuracy = (np.abs(predictions - labels) < value.tolerance).mean()

        return {"output": accuracy}

    def infer_types(self, input_types, value):
        return {"output": "float"}

    def infer_dims(self, input_dims, value):
        return {"output": 1}

class AccuracyValue:

    tolerance = 1e-8