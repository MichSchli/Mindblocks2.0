from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class TokenReplacerSequence(ComponentTypeModel):

    name = "TokenReplacer"
    in_sockets = ["target", "replacement"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return TokenReplacerValue(value_dictionary["token"][0][0])

    def execute(self, input_dictionary, value, output_value_models, mode):
        sequences = input_dictionary["target"].get_sequence()
        replacements = input_dictionary["replacement"].get_sequence()

        out = [None]*len(sequences)

        for i in range(len(sequences)):
            out[i] = [None]*len(sequences[i])
            for j in range(len(sequences[i])):
                if sequences[i][j] != value.target_token:
                    out[i][j] = sequences[i][j]
                else:
                    self.log("Replaced \"" + sequences[i][j] + "\" with \"" + replacements[i][j] + "\" at index " + str(j) + " in sentence \"" + " ".join(sequences[i]) + "\"", "formatting", "token_replacement")
                    out[i][j] = replacements[i][j]

        output_value_models["output"].assign(out)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["target"].copy()}

class TokenReplacerValue(ExecutionComponentValueModel):

    target_token = None

    def __init__(self, target_token):
        self.target_token = target_token