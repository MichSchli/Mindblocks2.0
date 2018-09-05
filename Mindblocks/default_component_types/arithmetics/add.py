from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class Add(ComponentTypeModel):

    name = "Add"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return AddValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        result = input_dictionary["left"].get_value() + input_dictionary["right"].get_value()
        output_value_models["output"].assign(result)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["left"].copy()}

class AddValue(ExecutionComponentValueModel):

    pass