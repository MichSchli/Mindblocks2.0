from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import numpy as np

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Concat(ComponentTypeModel):

    name = "Concat"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        return ConcatValue()

    def execute(self, input_dictionary, value, mode):
        # TODO: This is obviously wrong, but requires dim and typing refactoring
        return {"output": np.concatenate(([input_dictionary["left"]], [input_dictionary["right"]]))}

    def infer_types(self, input_types, value):
        return {"output": input_types["left"]}

    def infer_dims(self, input_dims, value):
        return {"output": input_dims["left"]}


class ConcatValue(ExecutionComponentValueModel):

    pass