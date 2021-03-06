from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class PassThrough(ComponentTypeModel):

    name = "PassThrough"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return PassThroughValue()

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        return {"output": input_dictionary["input"]}

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["input"].copy()}


class PassThroughValue(ExecutionComponentValueModel):

    pass