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
        if inp_val.is_value_type("list"):
            val = inp_val.get_value()
            #TODO: This is completely wrong

            out = []
            for i in range(len(val)):
                out.append([None]*len(val[i]))
                for j in range(len(val[i])):
                    out[i][j] = val[i][j][value.slices[2]]

            output_value_models["output"].assign_with_lengths(out, inp_val.get_lengths(), language="python")
        else:
            exit()
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()
        for i, dim_correction in enumerate(value.get_dim_corrections()):
            if dim_correction == "unknown":
                output_type.set_dimension(i, None)
            elif dim_correction == "singleton":
                output_type.delete_dimension(i)
            elif dim_correction is not None:
                output_type.set_dimension(i, dim_correction)

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
                    parts[0] = int(parts[0]) if parts[0] else 0
                    parts[1] = int(parts[1]) if parts[1] else -1

                    print(parts)

                    python_slices.append(slice(parts[0], 1, parts[1]))

                    if parts[0] < 0 or parts[1] < 0:
                        self.dim_corrections.append("unknown")
                    else:
                        self.dim_corrections.append(parts[1] - parts[0])


            self.slices = python_slices

        def get_slice_dims(self):
            return len(self.slices)

        def get_dim_corrections(self):
            return self.dim_corrections