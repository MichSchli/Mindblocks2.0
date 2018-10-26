import numpy as np
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class DataSlicer(ComponentTypeModel):

    name = "DataSlicer"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return DataSlicerValue(value_dictionary["slice"][0][0])

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        inp_val = input_dictionary["input"]
        if value.language == "python":
            val = inp_val.get_value()
            print(val[3])
            lengths = inp_val.get_lengths()[:]


            val = val[value.slices]

            deleted_dims = 0
            for i, dim_correction in enumerate(value.get_dim_corrections()):
                if dim_correction == "singleton":
                    del lengths[i - deleted_dims]
                    deleted_dims + 1

            output_value_models["output"].assign(val, length_list = lengths)
        else:
            exit()
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()
        deleted_dims = 0
        for i, dim_correction in enumerate(value.get_dim_corrections()):
            if dim_correction == "unknown":
                output_type.set_dimension(i - deleted_dims, None)
            elif dim_correction == "singleton":
                output_type.delete_dimension(i - deleted_dims)
                deleted_dims + 1
            elif dim_correction is not None:
                output_type.set_dimension(i - deleted_dims, dim_correction)

        return {"output": output_type}

class DataSlicerValue(ExecutionComponentValueModel):

        slices = None
        dim_corrections = None

        def __init__(self, slice_string):
            slice_parts = slice_string.split(",")
            python_slices = []
            self.dim_corrections = []
            for slice_part in slice_parts:
                if ":" not in slice_part:
                    python_slices.append(int(slice_part.strip()))
                    self.dim_corrections.append("singleton")
                else:
                    parts = [p.strip() for p in slice_part.split(":")]
                    parts[0] = int(parts[0]) if parts[0] else None
                    parts[1] = int(parts[1]) if parts[1] else None

                    python_slices.append(slice(parts[0], parts[1], 1))

                    firstdim_idx = parts[0] if parts[0] else 0
                    lastdim_idx = parts[1] if parts[1] else -1

                    if lastdim_idx < 0:
                        self.dim_corrections.append("unknown")
                    else:
                        self.dim_corrections.append(lastdim_idx - firstdim_idx)


            self.slices = python_slices

        def get_slice_dims(self):
            return len(self.slices)

        def get_dim_corrections(self):
            return self.dim_corrections