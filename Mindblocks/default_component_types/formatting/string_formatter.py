from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class StringFormatter(ComponentTypeModel):

    name = "StringFormatter"
    in_sockets = []
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return StringFormatterValue(value_dictionary["action"][0][0])

    def execute(self, input_dictionary, value, output_models, mode):
        string = value.action[:]
        out = []

        first_val = list(input_dictionary.values())[0].get_value()
        for i in range(len(first_val)):
            out.append([None] * len(first_val[i]))
            for j in range(len(first_val[i])):
                out[i][j] = value.action[:]

        for k,v in input_dictionary.items():
            for i in range(len(v.get_value())):
                for j in range(len(v.get_value()[i])):
                    out[i][j] = out[i][j].replace("["+k+"]", v.get_value()[i][j])

        output_models["output"].assign_with_lengths(out, list(input_dictionary.values())[0].get_lengths(), language="python")
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        ins = list(input_types.values())[0]
        return {"output": ins.copy()}


class StringFormatterValue(ExecutionComponentValueModel):

    def __init__(self, action):
        self.action = action