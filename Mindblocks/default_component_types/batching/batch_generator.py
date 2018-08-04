import random

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor_type import TensorType


class BatchGenerator(ComponentTypeModel):

    name = "BatchGenerator"
    in_sockets = ["count"]
    out_sockets = ["batch"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return BatchGeneratorValue(int(value_dictionary["batch_size"][0]))

    def execute(self, input_dictionary, value, mode):
        if value.needs_count():
            value.register_count(input_dictionary["count"])

        return {"batch": value.get_next_batch()}

    def build_value_type(self, input_types, value):
        return {"output": TensorType("int", value.batch_size)}

    def has_batches(self, value):
        return value.has_unyielded_batches()


class BatchGeneratorValue(ExecutionComponentValueModel):

    batch_size = None

    batches = None
    pointer = None

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def needs_count(self):
        return self.batches is None

    def register_count(self, count):
        indexes = list(range(count))
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