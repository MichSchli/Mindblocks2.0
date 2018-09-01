from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class StringFormatter(ComponentTypeModel):

    name = "StringFormatter"
    in_sockets = []
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return StringFormatterValue()

    def execute(self, input_dictionary, value, output_models, mode):
        return {"output": input_dictionary["input"]}

    def build_value_type_model(self, input_types, value, mode):
        ins = list(input_types.values())[0]
        return {"output": ins.copy()}


class StringFormatterValue(ExecutionComponentValueModel):

    pass