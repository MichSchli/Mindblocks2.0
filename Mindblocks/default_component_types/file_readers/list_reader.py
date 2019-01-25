import numpy as np

from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class ListReader(ComponentTypeModel):

    name = "ListReader"
    out_sockets = ["output", "count"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = ListReaderValue(value_dictionary["file_path"][0][0])

        if "batch_size" in value_dictionary:
            value.set_read_batches(int(value_dictionary["batch_size"][0][0]))

        separators = value_dictionary["separators"][0][0].split("|")
        value.set_separators(separators)

        soft_dim_symbol = "soft_dimensions"
        soft_dims = self.parse_dim_info(separators, soft_dim_symbol, value_dictionary)
        value.set_soft_dimensions(soft_dims)

        value.init_batches()

        return value

    def parse_dim_info(self, separators, info_symbol, value_dictionary):
        dims = [False] * len(separators)
        if info_symbol in value_dictionary:
            dims_to_set = value_dictionary[info_symbol][0][0].split(",")
            dims_to_set = [int(d) for d in dims_to_set]

            for d in dims_to_set:
                dims[d] = True
        return dims

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.use_read_batches():
            value.read_next_batch(serve=True)
        elif not value.has_read():
            value.read()

        as_tensor, length_list = value.as_soft_tensor()
        output_models["output"].assign(as_tensor, length_list)

        output_models["count"].assign(np.array(value.count()), length_list=None)
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        soft_dims = value.get_soft_by_dimensions()
        output_dims = value.infer_dims()

        output_type_model = SoftTensorTypeModel(output_dims, soft_by_dimensions=soft_dims, string_type="string")
        count_model = SoftTensorTypeModel([], string_type="int")

        return {"output": output_type_model,
                "count": count_model}

    def has_batches(self, value, previous_values, mode):
        has_batch = value.has_batch
        if not value.use_read_batches():
            value.has_batch = False
        return has_batch


class ListReaderValue(ExecutionComponentValueModel):

    filepath = None
    size = None
    separators = None
    full_list = None
    read_batches = None
    should_read_next_batch = True

    tensor = None

    f = None

    def __init__(self, filepath):
        self.filepath = filepath

    def init_batches(self):
        self.has_batch = True
        if self.use_read_batches():
            self.full_list = None
            self.should_read_next_batch = True
            if self.f is not None:
                self.f.close()
            self.f = open(self.filepath, 'r')

    def set_separators(self, separators):
        self.separators = separators

    def set_read_batches(self, batch_size):
        self.read_batches = batch_size

    def use_read_batches(self):
        return self.read_batches is not None

    def count(self):
        return self.size

    def process_recursively(self, text, separator_list):
        this_separator = separator_list[0]
        next_separators = separator_list[1:]

        if text == "":
            parts = []
        elif this_separator == "(C)":
            parts = text
        else:
            parts = text.split(this_separator)

        if len(next_separators) == 0:
            return parts
        else:
            return [self.process_recursively(part, next_separators) for part in parts]

    def set_soft_dimensions(self, soft_dims):
        self.soft_dims = soft_dims

    def get_soft_by_dimensions(self):
        return self.soft_dims

    def recursively_get_max_dim(self, this_l, level):
        remaining_levels = len(self.soft_dims) - level - 1

        if remaining_levels == 0:
            return [len(this_l)]
        else:
            this_level_dim = [len(this_l)]

            inner_dim_list = [self.recursively_get_max_dim(l, level + 1) for l in this_l]
            largest = [0] * remaining_levels

            for j in range(remaining_levels):
                level_list = [inner[j] for inner in inner_dim_list]
                largest[j] = max(level_list) if len(level_list) > 0 else 0

            return this_level_dim + largest

    def read_next_batch(self, serve=True):
        if self.should_read_next_batch:
            full_text = ""

            counter = 0
            next_line = self.f.readline()

            while counter < self.read_batches and next_line:
                full_text += next_line
                if full_text.endswith(self.separators[0]):
                    counter += 1

                if counter < self.read_batches:
                    next_line = self.f.readline()

            # If we stopped because of the batch, delete a trailing separator:
            if counter == self.read_batches:
                full_text = full_text[:-len(self.separators[0])]
            else:
                self.has_batch = False
                self.f.close()
                self.f = None

            processed = self.process_recursively(full_text, self.separators)
            self.full_list = processed

        self.should_read_next_batch = serve

    def infer_dims(self):
        if self.use_read_batches() and self.full_list is None:
            self.read_next_batch(serve=False)
        elif not self.has_read():
            self.read()

        dims = self.recursively_get_max_dim(self.full_list, 0)

        if self.use_read_batches():
            dims[0] = None
            for d in self.soft_dims:
                dims[d] = None

        return dims

    def read(self):
        if self.full_list is None:
            full_text = ""

            f = open(self.filepath, 'r')
            for line in f:
                full_text += line

            f.close()

            processed = self.process_recursively(full_text, self.separators)
            self.full_list = processed

        return self.full_list

    def has_read(self):
        return self.full_list is not None

    def as_soft_tensor(self):
        if self.tensor is None or self.use_read_batches():
            sth = SoftTensorHelper()
            self.tensor, self.length_list = sth.to_soft_tensor(self.full_list, self.infer_dims(), self.get_soft_by_dimensions(), "string")

        return self.tensor, self.length_list
