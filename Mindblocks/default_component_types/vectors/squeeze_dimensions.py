from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf
import numpy as np

class SqueezeDimensions(ComponentTypeModel):

    name = "SqueezeDimensions"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = SqueezeDimensionsValue()
        value.language = language

        dims = value_dictionary["dims"][0][0].split(",")
        dims = [int(x) for x in dims]

        value.set_dims(dims)

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.language == "python":
            dims = tuple(value.get_dims())

            val = input_dictionary["input"].get_value()
            val = np.squeeze(val, axis=dims)

            lengths = input_dictionary["input"].get_lengths()[:]

            offset = 0
            for d in dims:
                del lengths[d - offset]

            output_models["output"].assign(val, lengths)

        elif value.language == "tensorflow":
            print("didnt implement squeeze_dimension for tensorflow")
            exit()
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        out_type = input_types["input"].copy()

        for idx in value.get_dims():
            out_type.delete_dimension(idx)

        return {"output": out_type}


class SqueezeDimensionsValue(ExecutionComponentValueModel):

    def set_dims(self, dims):
        self.dims = dims

    def get_dims(self):
        return self.dims