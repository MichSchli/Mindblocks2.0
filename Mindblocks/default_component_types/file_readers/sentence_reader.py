from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class SentenceReader(ComponentTypeModel):

    name = "SentenceReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = SentenceReaderValue(value_dictionary["file_path"][0][0])

        if "start_token" in value_dictionary:
            value.set_start_token(value_dictionary["start_token"][0][0])
        if "stop_token" in value_dictionary:
            value.set_stop_token(value_dictionary["stop_token"][0][0])

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        output_models["output"].initial_assign(value.read())
        output_models["count"].assign(value.count(), length_list=None)
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        num_examples = len(value.read())

        output_dims = [num_examples, None]
        soft_dims = [False, True]

        output_type_model = SoftTensorTypeModel(output_dims, soft_by_dimensions=soft_dims, string_type="string")
        count_model = SoftTensorTypeModel([], string_type="int")

        return {"output": output_type_model,
                "count": count_model}

    def has_batches(self, value, previous_values, mode):
        has_batch = value.has_batch
        value.has_batch = False
        return has_batch


class SentenceReaderValue(ExecutionComponentValueModel):

    filepath = None
    size = None
    start_token = None
    stop_token = None

    def __init__(self, filepath):
        self.filepath = filepath
        self.has_batch = True

    def set_start_token(self, token):
        self.start_token = token

    def set_stop_token(self, token):
        self.stop_token = token

    def get_start_token_part(self):
        return self.start_token

    def get_stop_token_part(self):
        return self.stop_token

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
                lines[-1].extend(line.split(" "))

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