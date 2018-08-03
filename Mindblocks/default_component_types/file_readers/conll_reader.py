from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class ConllReader(ComponentTypeModel):

    name = "ConllReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return ConllReaderValue(value_dictionary["file_path"][0],
                                value_dictionary["columns"][0].split(","))

    def execute(self, input_dictionary, value, mode):
        return {"output": value.read(), "count": value.count()}

    def infer_types(self, input_types, value):
        return {"output": "sequence", "count": "int"}

    def infer_dims(self, input_dims, value):
        return {"output": [None, None, value.count_columns()], "count": 1}

class ConllReaderValue(ExecutionComponentValueModel):

    filepath = None
    size = None

    def __init__(self, filepath, column_info):
        self.filepath = filepath
        self.column_info = column_info

    def count(self):
        return self.size

    def read(self):
        lines = [[]]
        f = open(self.filepath, 'r')
        for line in f:
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

        f.close()

        return lines