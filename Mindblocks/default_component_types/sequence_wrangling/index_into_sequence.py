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
        sequences = input_dictionary["sequence"].get_value()
        indexes = input_dictionary["indexes"].get_value()

        index_lengths = input_dictionary["indexes"].get_lengths()

        print(indexes)
        print(index_lengths)

        index_specific_lengths = index_lengths[1]

        out = [None]*indexes.shape[0]

        for i in range(indexes.shape[0]):
            out[i] = [None]*index_specific_lengths[i]
            for j in range(index_specific_lengths[i]):
                out[i][j] = sequences[i][indexes[i][j]]

        output_value_models["output"].initial_assign(out)

        print(output_value_models["output"].get_dimensions())

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["indexes"].copy()

        output_type.set_data_type(input_types["sequence"].get_data_type())
        output_type.set_dimension(-1, input_types["sequence"].get_dimension(-1))

        print(input_types["indexes"].get_dimensions())
        print(input_types["sequence"].get_dimensions())
        print(output_type.get_dimensions())

        return {"output": output_type}


class IndexIntoSequenceValue(ExecutionComponentValueModel):

    pass