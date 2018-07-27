import random

import numpy as np
from model.component_type.component_type_model import ComponentTypeModel


class Batcher(ComponentTypeModel):

    name = "Batcher"
    in_sockets = ["data", "indexes"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return BatcherValue(value_dictionary["lazy"] == "True")

    def execute(self, input_dictionary, value, mode):
        data = np.array(input_dictionary["data"])
        indexes = input_dictionary["indexes"]

        if value.lazy:
            return {"output": data[indexes]}

        return None

    def infer_types(self, input_types, value):
        return {"output": "string"}

    def infer_dims(self, input_dims, value):
        return {"output": [None, None]}


class BatcherValue:

    lazy = None

    def __init__(self, lazy):
        self.lazy = lazy