from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.old.sequence_batch_type import SequenceBatchType
from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel


class DeIndexer(ComponentTypeModel):
    name = "DeIndexer"
    in_sockets = ["input", "index"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return DeIndexerValue(value_dictionary["input_type"][0])

    def execute(self, input_dictionary, value, output_models, mode):
        transformed_input = value.apply_index(input_dictionary["input"].get_sequence(),
                                              input_dictionary["index"].get_index())

        output_models["output"].assign(transformed_input)

        return output_models

    def build_value_type_model(self, input_types, value):
        return {"output": SequenceBatchTypeModel("string", [], input_types["input"].get_batch_size())}


class DeIndexerValue(ExecutionComponentValueModel):

    def __init__(self, input_type):
        self.input_type = input_type

    def apply_index(self, input_value, index):
        if self.input_type == "sequence":
            for i in range(len(input_value)):
                for j in range(len(input_value[i])):
                    to_index = input_value[i][j]

                    if to_index not in index["forward"]:
                        index["forward"][to_index] = len(index["forward"]) + 1
                        index["backward"][index["forward"][to_index]] = to_index

                    input_value[i][j] = index["backward"][to_index]

        return input_value