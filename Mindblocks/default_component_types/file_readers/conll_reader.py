from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.old.sequence_batch_type import SequenceBatchType
from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class ConllReader(ComponentTypeModel):

    name = "ConllReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return ConllReaderValue(value_dictionary["file_path"][0][0],
                                value_dictionary["columns"][0][0].split(","))

    def execute(self, input_dictionary, value, output_models, mode):
        output_models["output"].assign(value.read())
        output_models["count"].assign(value.count)
        return output_models

    def build_value_type_model(self, input_types, value):
        return {"output": SequenceBatchTypeModel("string", [value.count_columns()], None),
                "count": TensorTypeModel("int", [])}


class ConllReaderValue(ExecutionComponentValueModel):

    filepath = None
    size = None

    def __init__(self, filepath, column_info):
        self.filepath = filepath
        self.column_info = column_info

    def count(self):
        return self.size

    def count_columns(self):
        return len(self.column_info)

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