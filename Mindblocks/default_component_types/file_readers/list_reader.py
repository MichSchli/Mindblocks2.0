from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.list_batch.list_batch_type_model import ListBatchTypeModel
from Mindblocks.model.value_type.old.sequence_batch_type import SequenceBatchType
from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class ListReader(ComponentTypeModel):

    name = "ListReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = ListReaderValue(value_dictionary["file_path"][0][0])

        if "separator" in value_dictionary:
            value.set_separator(value_dictionary["separator"][0][0])

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        output_models["output"].assign(value.read())
        output_models["count"].assign(value.count())
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        read_value = value.read()

        return {"output": ListBatchTypeModel("string", [len(read_value[0][0])], len(read_value), max([len(v) for v in read_value])),
                "count": TensorTypeModel("int", [])}

    def has_batches(self, value, previous_values, mode):
        has_batch = value.has_batch
        value.has_batch = False
        return has_batch

class ListReaderValue(ExecutionComponentValueModel):

    filepath = None
    size = None
    separator = None
    lines = None

    def __init__(self, filepath):
        self.filepath = filepath
        self.has_batch = True
        self.separator = ","

    def set_separator(self, separator):
        self.separator = separator.replace("\\t", "\t")

    def init_batches(self):
        self.has_batch = True

    def count(self):
        return self.size

    def read(self):
        if self.lines is not None:
            return self.lines
        lines = [[]]
        f = open(self.filepath, 'r')
        for line in f:
            line = line.strip()

            if line:
                lines[-1].append(line.split(self.separator))
            else:
                lines.append([])

        if not lines[-1]:
            lines = lines[:-1]

        self.size = len(lines)
        self.lines = lines

        f.close()

        return lines