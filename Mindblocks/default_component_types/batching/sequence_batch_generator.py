import random

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class BatchGenerator(ComponentTypeModel):

    name = "SequenceBatchGenerator"
    in_sockets = ["reference_sequences"]
    out_sockets = ["batch"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = SequenceBatchGeneratorValue(int(value_dictionary["batch_size"][0][0]))

        if "shuffle" in value_dictionary:
            value.set_shuffle(value_dictionary["shuffle"][0][0] == "True")

        if "reorder" in value_dictionary:
            value.set_should_reorder(value_dictionary["reorder"][0][0] == "True")

        if "prefer_reduce" in value_dictionary:
            should_prefer_reduce = value_dictionary["prefer_reduce"][0][0] == "True"
            value.set_prefer_reduce(should_prefer_reduce)

        return value

    def execute(self, input_dictionary, value, output_value_models, mode):
        if value.should_initialize(mode):
            value.register_lengths(input_dictionary["reference_sequences"].get_sequence_lengths(), mode)

        if value.should_shuffle_now(mode):
            value.shuffle(mode)

        output_value_models["batch"].assign(value.get_next_batch(mode))
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"batch": TensorTypeModel("int", [value.batch_size])}

    def has_batches(self, value, previous_values, mode):
        return value.has_unyielded_batches(mode)


class SequenceBatchGeneratorValue(ExecutionComponentValueModel):

    batch_size = None

    batches = None
    pointer = None
    should_shuffle = None
    should_reorder = None
    prefer_reduce = None

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.should_shuffle = True
        self.batches = {"train": None, "validate": None, "test": None}
        self.pointer = 0
        self.should_reorder = True

    def set_shuffle(self, should_shuffle):
        self.should_shuffle = should_shuffle

    def set_should_reorder(self, should_reorder):
        self.should_reorder = should_reorder

    def set_prefer_reduce(self, prefer_reduce):
        self.prefer_reduce = prefer_reduce

    def should_initialize(self, mode):
        return self.batches[mode] is None

    def register_lengths(self, lengths, mode):
        self.log("Creating batches with mode " + mode + "...", "batching", "status")
        if not self.should_reorder:
            self.batches[mode] = []
            for i in range(0, len(lengths), self.batch_size):
                batch = list(range(i, min(i+self.batch_size, len(lengths))))
                self.batches[mode].append(batch)

            self.log("Created " + str(len(self.batches[mode])) + " batches.", "batching", "status")

            return

        indexes = list(range(len(lengths)))

        if self.should_shuffle and mode == "train":
            random.shuffle(indexes)

        reordered_lengths = [lengths[i] for i in indexes]
        indexes_by_size = [x for _,x in sorted(zip(reordered_lengths, indexes), key=lambda pair: pair[0])]
        lengths_by_size = [lengths[i] for i in indexes_by_size]

        self.batches[mode] = [[]]
        length_tracker = None
        if self.prefer_reduce:
            for i in range(len(indexes_by_size)):
                current_length = lengths_by_size[i]
                if len(self.batches[mode][-1]) == self.batch_size or (len(self.batches[mode][-1]) > 0 and current_length > length_tracker):
                    self.log("Created batch with " + str(len(self.batches[mode][-1])) + " sequences of length " + str(length_tracker) + ".", "batching", "update")
                    self.batches[mode].append([])

                self.batches[mode][-1].append(indexes_by_size[i])
                length_tracker = current_length

        self.log("Created " + str(len(self.batches[mode])) + " batches.", "batching", "status")
        self.pointer = 0

    def should_shuffle_now(self, mode):
        return mode == "train" and self.pointer == 0

    def shuffle(self, mode):
        random.shuffle(self.batches[mode])

    def get_next_batch(self, mode):
        batch = self.batches[mode][self.pointer]

        self.pointer += 1

        return batch

    def init_batches(self):
        self.pointer = 0

    def has_unyielded_batches(self, mode):
        return self.batches[mode] is None or self.pointer < len(self.batches[mode])