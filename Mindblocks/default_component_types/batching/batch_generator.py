import random

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class BatchGenerator(ComponentTypeModel):

    name = "BatchGenerator"
    in_sockets = ["count"]
    out_sockets = ["batch"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = BatchGeneratorValue(int(value_dictionary["batch_size"][0][0]))

        if "shuffle" in value_dictionary:
            value.set_shuffle(value_dictionary["shuffle"][0][0] == "True")

        return value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        if value.needs_count():
            value.register_count(input_dictionary["count"].get_value(), mode)

        output_value_models["batch"].assign(value.get_next_batch())
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        index_tensor_type = SoftTensorTypeModel([value.batch_size],
                                                 string_type="int")
        return {"batch": index_tensor_type}

    def has_batches(self, value, previous_values, mode):
        return value.has_unyielded_batches()


class BatchGeneratorValue(ExecutionComponentValueModel):

    batch_size = None

    batches = None
    pointer = None
    should_shuffle = None

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.should_shuffle = True

    def needs_count(self):
        return self.batches is None

    def set_shuffle(self, should_shuffle):
        self.should_shuffle = should_shuffle

    def register_count(self, count, mode):
        indexes = list(range(count))
        if self.should_shuffle and mode == "train":
            random.shuffle(indexes)

        self.batches = indexes
        self.pointer = 0

    def get_next_batch(self):
        batch = self.batches[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size

        return batch

    def init_batches(self):
        self.batches = None
        self.pointer = None

    def has_unyielded_batches(self):
        return self.batches is None or self.pointer < len(self.batches)

    def get_batch_size(self):
        return self.batch_size