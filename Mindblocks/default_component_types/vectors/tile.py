from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf
import numpy as np

class Tile(ComponentTypeModel):

    name = "Tile"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = TileValue()
        value.language = language

        dim_changes = value_dictionary["dim_changes"][0][0].split(",")
        dim_changes = [c.split(":") for c in dim_changes]
        dim_changes = [[int(x[0]), int(x[1])] for x in dim_changes]

        value.set_dim_changes(dim_changes)

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.language == "python":
            v = input_dictionary["input"].get_value()
            lengths = input_dictionary["input"].get_lengths()


            for idx, factor in value.get_dim_changes():
                v = np.repeat(v, factor, axis=idx)

                for length_idx in range(0, idx + 1):
                    if lengths[length_idx] is not None:
                        lengths[length_idx] = np.repeat(lengths[length_idx], factor, axis=idx)

            output_models["output"].assign(v, length_list=lengths)
        elif value.language == "tensorflow":
            print("didnt implement add_dimension for tensorflow")
            exit()

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        out_type = input_types["input"].copy()
        prev_dims = out_type.get_dimensions()

        for idx, factor in value.get_dim_changes():
            new_dimension = None if prev_dims[idx] is None else factor * prev_dims[idx]
            out_type.set_dimension(idx, new_dimension)

        return {"output": out_type}


class TileValue(ExecutionComponentValueModel):

    def set_dim_changes(self, dim_changes):
        self.dim_changes = dim_changes

    def get_dim_changes(self):
        return self.dim_changes