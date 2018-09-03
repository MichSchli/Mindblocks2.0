from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.old.sequence_batch_type import SequenceBatchType
from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class ConllReader(ComponentTypeModel):

    name = "ConllReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = ConllReaderValue(value_dictionary["file_path"][0][0],
                                value_dictionary["columns"][0][0].split(","))

        if "start_token" in value_dictionary:
            value.set_start_token(value_dictionary["start_token"][0][0])
        if "stop_token" in value_dictionary:
            value.set_stop_token(value_dictionary["stop_token"][0][0])

        return value

    def execute(self, input_dictionary, value, output_models, mode):
        output_models["output"].assign(value.read())
        output_models["count"].assign(value.count())
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": SequenceBatchTypeModel("string", [value.count_columns()], len(value.read()), max([len(v) for v in value.read()])),
                "count": TensorTypeModel("int", [])}

    def has_batches(self, value, previous_values, mode):
        has_batch = value.has_batch
        value.has_batch = False
        return has_batch


class ConllReaderValue(ExecutionComponentValueModel):

    filepath = None
    size = None
    start_token = None
    stop_token = None

    def __init__(self, filepath, column_info):
        self.filepath = filepath
        self.column_info = column_info
        self.has_batch = True

    def set_start_token(self, token):
        self.start_token = token

    def set_stop_token(self, token):
        self.stop_token = token

    def get_start_token_part(self):
        parts = ["_" for _ in range(self.count_columns())]
        parts[1] = self.start_token
        return parts

    def get_stop_token_part(self):
        parts = ["_" for _ in range(self.count_columns())]
        parts[1] = self.stop_token
        return parts

    def init_batches(self):
        self.has_batch = True

    def count(self):
        return self.size

    def count_columns(self):
        return len(self.column_info)

    def read(self):
        lines = [[]] if self.start_token is None else [[self.get_start_token_part()]]
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
                if self.stop_token is not None:
                    lines[-1].append(self.get_stop_token_part())

                if self.start_token is not None:
                    lines.append([self.get_start_token_part()])
                else:
                    lines.append([])

        if not lines[-1] or lines[-1] == [self.get_start_token_part()]:
            lines = lines[:-1]

        if self.stop_token is not None and len(lines) > 0 and lines[-1][-1] != self.get_stop_token_part():
            lines[-1].append(self.get_stop_token_part())

        self.size = len(lines)

        f.close()

        return lines
