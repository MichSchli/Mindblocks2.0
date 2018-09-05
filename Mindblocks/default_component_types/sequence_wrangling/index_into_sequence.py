from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class IndexIntoSequence(ComponentTypeModel):

    name = "IndexIntoSequence"
    in_sockets = ["indexes", "sequence"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return IndexIntoSequenceValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        sequences = input_dictionary["sequence"].get_sequence()
        indexes = input_dictionary["indexes"].get_sequence()

        out = [None]*len(sequences)

        for i in range(len(sequences)):
            out[i] = [None]*len(indexes[i])
            for j in range(len(indexes[i])):
                out[i][j] = sequences[i][indexes[i][j]]

        output_value_models["output"].assign(out)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["indexes"].copy()

        output_type.set_data_type(input_types["sequence"].get_data_type())
        output_type.extend_dims(input_types["sequence"].get_inner_dim())

        return {"output": output_type}


class IndexIntoSequenceValue(ExecutionComponentValueModel):

    pass