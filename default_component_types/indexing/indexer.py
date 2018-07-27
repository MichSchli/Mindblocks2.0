from model.component_type.component_type_model import ComponentTypeModel


class Indexer(ComponentTypeModel):
    name = "Indexer"
    in_sockets = ["input", "index"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return IndexerValue(value_dictionary["input_type"],
                            int(value_dictionary["input_column"]))

    def execute(self, input_dictionary, value, mode):
        transformed_input = value.apply_index(input_dictionary["input"],
                                              input_dictionary["index"])

        return {"output": transformed_input}

    def infer_types(self, input_types, value):
        return {"output": "int"}

    def infer_dims(self, input_dims, value):
        return {"output": input_dims["input"]}


class IndexerValue:

    def __init__(self, input_type, input_column):
        self.input_type = input_type
        self.input_column = input_column

    def apply_index(self, input_value, index):
        if self.input_type == "sequence":
            for i in range(len(input_value)):
                for j in range(len(input_value[i])):
                    to_index = input_value[i][j][self.input_column]

                    if to_index not in index["forward"]:
                        index["forward"][to_index] = len(index["forward"])
                        index["backward"][index["forward"][to_index]] = to_index

                    input_value[i][j][self.input_column] = index["forward"][to_index]
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