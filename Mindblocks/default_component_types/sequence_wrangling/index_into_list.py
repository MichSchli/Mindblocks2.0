from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class IndexIntoList(ComponentTypeModel):

    name = "IndexIntoList"
    in_sockets = ["indexes", "list"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return IndexIntoListValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        sequences = input_dictionary["list"].get_value()
        indexes = input_dictionary["indexes"].get_value()

        out = [None]*len(sequences)

        for i in range(len(sequences)):
            #TODO this should be handled in preprocessing but I'm lazy
            if len(sequences[i]) == 0:
                out[i] = None
            else:
                out[i] = sequences[i][indexes[i]]

        output_value_models["output"].assign(out)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["indexes"].copy()

        output_type.set_data_type(input_types["list"].data_type)
        output_type.extend_dims(input_types["list"].get_inner_dim())

        return {"output": output_type}

class IndexIntoListValue(ExecutionComponentValueModel):

    pass