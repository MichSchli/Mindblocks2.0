from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class AddDimensions(ComponentTypeModel):

    name = "AddDimensions"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = AddDimensionsValue()
        value.language = language

        dim_changes = value_dictionary["dim_changes"][0][0].split(",")
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

            dim_update = [1] * len(lengths)
            dim_not_1 = False

            for idx, dims_to_add in value.get_dim_changes():
                v = tf.expand_dims(v, idx)

                if idx == -1:
                    lengths.append(None)
                elif idx < 0:
                    lengths.insert(idx + 1, None)
                else:
                    lengths.insert(idx, None)

                iter_point = idx if idx > 0 else len(lengths) - idx

                for l in range(iter_point, len(lengths)):
                    if lengths[l] is not None:
                        lengths[l] = tf.expand_dims(lengths[l], idx)

                        if dims_to_add > 1:
                            length_expansion = [1] * (idx + 1)
                            length_expansion[l] *= dims_to_add
                            lengths[l] = tf.tile(lengths[l], length_expansion)

                dim_update.insert(idx, dims_to_add)

                if dims_to_add > 1:
                    dim_not_1 = True

            if dim_not_1:
                v = tf.tile(v, dim_update)

            output_models["output"].assign(v, length_list=lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        out_type = input_types["input"].copy()

        for idx, dims_to_add in value.get_dim_changes():
            out_type.add_dimension(idx, dims_to_add)

        return {"output": out_type}


class AddDimensionsValue(ExecutionComponentValueModel):

    def set_dim_changes(self, dim_changes):
        self.dim_changes = dim_changes

    def get_dim_changes(self):
        return self.dim_changes