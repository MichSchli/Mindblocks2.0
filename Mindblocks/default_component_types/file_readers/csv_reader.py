from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class CsvReader(ComponentTypeModel):

    name = "CsvReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return CsvReaderValue(value_dictionary["file_path"][0][0],
                              value_dictionary["columns"][0][0].split(","))

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        output_value_models["output"].initial_assign(value.read())
        output_value_models["count"].initial_assign(value.count())
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_tensor_type = SoftTensorTypeModel([None, value.count_columns()],
                                                 string_type="string")

        count_tensor_type = SoftTensorTypeModel([],
                                                 string_type="int")

        return {"output": output_tensor_type,
                "count": count_tensor_type}


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