from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class ModeSplitter(ComponentTypeModel):

    name = "ModeSplitter"
    in_sockets = ["train", "validate", "test"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ModeSplitterValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        result = input_dictionary[mode].get_value()
        lengths = input_dictionary[mode].get_lengths()
        output_value_models["output"].assign(result, lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types[mode].copy()}

    def is_used(self, socket_name, mode):
        return socket_name == mode

class ModeSplitterValue(ExecutionComponentValueModel):

    pass