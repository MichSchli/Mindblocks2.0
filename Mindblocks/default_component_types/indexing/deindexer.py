from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class DeIndexer(ComponentTypeModel):
    name = "DeIndexer"
    in_sockets = ["input", "index"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        de_indexer_value = DeIndexerValue(value_dictionary["input_type"][0][0])

        return de_indexer_value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        transformed_input = value.apply_index(input_dictionary["input"].get_value(),
                                              input_dictionary["index"].get_index())

        output_models["output"].assign(transformed_input, length_list=input_dictionary["input"].get_lengths())

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["input"].copy()
        output_type.set_data_type("string")
        return {"output": output_type}


class DeIndexerValue(ExecutionComponentValueModel):

    def __init__(self, input_type):
        self.input_type = input_type

    def apply_index(self, input_value, index):
        new_output = []
        if self.input_type == "sequence":
            for i in range(len(input_value)):
                new_output.append([])
                for j in range(len(input_value[i])):
                    to_index = input_value[i][j]

                    new_output[i].append(index["backward"][to_index])

        return new_output