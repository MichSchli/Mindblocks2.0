from Mindblocks.model.component_type.component_type_model import ComponentTypeModel


class CsvReader(ComponentTypeModel):

    name = "CsvReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return CsvReaderValue(value_dictionary["file_path"],
                              value_dictionary["columns"].split(","))

    def execute(self, input_dictionary, value, mode):
        return {"output": value.read(), "count": value.count()}

    def infer_types(self, input_types, value):
        return {"output": value.column_info, "count": "int"}

    def infer_dims(self, input_dims, value):
        return {"output": [None, value.count_columns()], "count": 1}


class CsvReaderValue:

    filepath = None
    separator = ","
    size = None

    def __init__(self, filepath, column_info):
        self.filepath = filepath
        self.column_info = column_info

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