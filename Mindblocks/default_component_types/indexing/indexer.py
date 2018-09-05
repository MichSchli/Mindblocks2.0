from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np


class Indexer(ComponentTypeModel):
    name = "Indexer"
    in_sockets = ["input", "index"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        indexer_value = IndexerValue(value_dictionary["input_type"][0][0])

        if "input_column" in value_dictionary:
            indexer_value.set_input_column(int(value_dictionary["input_column"][0][0]))

        return indexer_value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        transformed_input = value.apply_index(input_dictionary["input"].get_value(),
                                              input_dictionary["index"].get_index())

        output_value_models["output"].assign(transformed_input)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        input_value_type = input_types["input"].copy()
        input_value_type.set_data_type("int")
        input_value_type.set_inner_dim(1)
        return {"output": input_value_type}


class IndexerValue(ExecutionComponentValueModel):

    input_column = None

    def __init__(self, input_type):
        self.input_type = input_type
        self.input_column = None

    def set_input_column(self, column_index):
        self.input_column = column_index

    def apply_index(self, input_value, index):
        if self.input_type == "sequence" or self.input_type == "list":
            output = []
            for i in range(len(input_value)):
                output.append([])
                for j in range(len(input_value[i])):
                    if self.input_column is not None:
                        to_index = input_value[i][j][self.input_column]
                    else:
                        to_index = input_value[i][j]

                    if to_index not in index["forward"]:
                        if "unk_token" in index and index["unk_token"] is not None:
                            to_index = index["unk_token"]
                        else:
                            index["forward"][to_index] = len(index["forward"])
                            index["backward"][index["forward"][to_index]] = to_index

                    output[i].append(index["forward"][to_index])

            return output
        elif self.input_type.startswith("tensor"):
            total_dims = int(self.input_type.split(":")[1])
            if total_dims == 2:
                for i in range(len(input_value)):
                    to_index = input_value[i][self.input_column]

                    if to_index not in index["forward"]:
                        index["forward"][to_index] = len(index["forward"])
                        index["backward"][index["forward"][to_index]] = to_index

                    input_value[i][self.input_column] = index["forward"][to_index]

        return input_value