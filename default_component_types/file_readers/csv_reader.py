from model.component_type.component_type_model import ComponentTypeModel


class CsvReader(ComponentTypeModel):

    name = "CsvReader"
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return CsvReaderValue(value_dictionary["file_path"],
                              value_dictionary["columns"].split(","))

    def execute(self, input_dictionary, value):
        return {"output": value.read()}

    def infer_types(self, input_types, value):
        return {"output": value.column_info}

    def infer_dims(self, input_dims, value):
        return {"output": [None, value.count_columns()]}


class CsvReaderValue:

    filepath = None
    separator = ","

    def __init__(self, filepath, column_info):
        self.filepath = filepath
        self.column_info = column_info

    def read(self):
        lines = []
        for line in open(self.filepath, 'r'):
            line = line.strip()

            if line:
                line_parts = line.split(self.separator)

                for i, column_type in enumerate(self.column_info):
                    if column_type == "int":
                        line_parts[i] = int(line_parts[i])

                lines.append(line_parts)

        return lines