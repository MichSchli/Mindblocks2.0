from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class SliceComponent(ComponentTypeModel):

    name = "Slice"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return SliceValue(value_dictionary["slice"][0][0])

    def execute(self, input_dictionary, value, output_models, mode):
        return {"output": input_dictionary["input"][value.slices]}

    def build_value_type_model(self, input_types, value):
        output_type = input_types["input"].copy()
        for i, dim_correction in enumerate(value.dim_corrections()):
            if dim_correction == "unknown":
                output_type.set_dim(i, None)
            elif dim_correction == "singleton":
                output_type.remove_dim(i)
            elif dim_correction is not None:
                output_type.set_dim(i, dim_correction)
        return {"output": output_type}


class SliceValue(ExecutionComponentValueModel):

    slices = None

    def __init__(self, slice_string):
        slice_parts = slice_string.split(",")
        python_slices = []
        for slice_part in slice_parts:
            if ":" not in slice_part:
                python_slices.append(slice(int(slice_part.strip())))
            else:
                parts = [p.strip() for p in slice_part.split(":")]
                parts[0] = int(parts[0]) if parts[0] else 0
                parts[1] = int(parts[1]) if parts[1] else -1

                python_slices.append(slice(parts[0], 1, parts[1]))

        self.slices = python_slices

    def get_slice_dims(self):
        return len(self.slices)