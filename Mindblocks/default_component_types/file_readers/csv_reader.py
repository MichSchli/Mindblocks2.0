from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.old.tensor_type import TensorType
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class CsvReader(ComponentTypeModel):

    name = "CsvReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return CsvReaderValue(value_dictionary["file_path"][0],
                              value_dictionary["columns"][0].split(","))

    def execute(self, input_dictionary, value, output_value_models, mode):
        output_value_models["output"].assign(value.read())
        output_value_models["count"].assign(value.count())
        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"output": TensorTypeModel("string", [None, value.count_columns()]),
                "count": TensorTypeModel("int", [])}


class CsvReaderValue(ExecutionComponentValueModel):

    filepath = None
    separator = ","
    size = None

    def __init__(self, filepath, column_info):
        self.filepath = filepath
        self.column_info = column_info

    def count_columns(self):
        return len(self.column_info)

    def read(self):
        lines = []
        f = open(self.filepath, 'r')
        for line in f:
            line = line.strip()

            if line:
                line_parts = line.split(self.separator)

                for i, column_type in enumerate(self.column_info):
                    if column_type == "int":
                        line_parts[i] = int(line_parts[i])

                lines.append(line_parts)

        self.size = len(lines)
        f.close()

        return lines

    def count_columns(self):
        return len(self.column_info)

    def count(self):
        return self.size