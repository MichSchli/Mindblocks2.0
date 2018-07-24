from model.component_type.component_type_model import ComponentTypeModel


class ConllReader(ComponentTypeModel):

    name = "ConllReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return ConllReaderValue(value_dictionary["file_path"],
                                value_dictionary["columns"].split(","))

    def execute(self, input_dictionary, value):
        return {"output": value.read(), "count": value.count()}

    def infer_types(self, input_types, value):
        return {"output": "sequence", "count": "int"}

    def infer_dims(self, input_dims, value):
        return {"output": [None, None, value.count_columns()], "count": 1}

class ConllReaderValue:

    filepath = None
    size = None

    def __init__(self, filepath, column_info):
        self.filepath = filepath
        self.column_info = column_info

    def count(self):
        return self.size

    def read(self):
        lines = [[]]
        for line in open(self.filepath, 'r'):
            line = line.strip()

            if line:
                line_parts = line.split('\t')

                for i, column_type in enumerate(self.column_info):
                    if column_type == "int":
                        line_parts[i] = int(line_parts[i])

                lines[-1].append(line_parts)
            else:
                lines.append([])

        if not lines[-1]:
            lines = lines[-1]

        self.size = len(lines)

        return lines