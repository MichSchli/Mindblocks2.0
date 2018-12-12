from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class Pad(ComponentTypeModel):

    name = "Pad"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = PadValue()
        value.language = language

        dim_changes = value_dictionary["pad_dimensions"][0][0].split(",")
        dim_changes = [c.split(":") for c in dim_changes]
        dim_changes = [[int(x[0]), int(x[1])] for x in dim_changes]

        value.set_dim_changes(dim_changes)

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.language == "python":
            print("didnt implement add_dimension for python")
            exit()
        elif value.language == "tensorflow":
            v = input_dictionary["input"].get_value()
            lengths = input_dictionary["input"].get_lengths()[:]

            paddings = [[0,0] for _ in lengths]

            for idx, dims_to_add in value.get_dim_changes():
                current_size = input_dictionary["input"].get_dimensions()[idx]
                required_padding = dims_to_add - current_size

                paddings[idx][1] = required_padding

            v = tf.pad(v, paddings, "CONSTANT", constant_values=-10)

            output_models["output"].assign(v, length_list=lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        out_type = input_types["input"].copy()

        for idx, dims_to_add in value.get_dim_changes():
            out_type.set_dimension(idx, dims_to_add)

        return {"output": out_type}


class PadValue(ExecutionComponentValueModel):

    def set_dim_changes(self, dim_changes):
        self.dim_changes = dim_changes

    def get_dim_changes(self):
        return self.dim_changes