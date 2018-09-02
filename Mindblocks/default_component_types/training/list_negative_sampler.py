from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class ListNegativeSampler(ComponentTypeModel):

    name = "ListNegativeSampler"
    in_sockets = ["list"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return ListNegativeSamplerValue()

    def execute(self, input_dictionary, value, output_models, mode):
        return {"output": input_dictionary["input"]}

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["list"].copy()}


class ListNegativeSamplerValue(ExecutionComponentValueModel):

    pass