from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class DebugPrint(ComponentTypeModel):

    name = "DebugPrint"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = DebugPrintValue()
        value.language = language
        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.language == "python":
            print(input_dictionary["input"].get_value())
        elif value.language == "tensorflow":
            v = input_dictionary["input"].get_value()
            v = tf.Print(v, [v], message="debug", summarize=100)
            input_dictionary["input"].assign(v, language="tensorflow")
        return {"output": input_dictionary["input"]}

    def build_value_type_model(self, input_types, value, mode):
        return {"output": input_types["input"].copy()}


class DebugPrintValue(ExecutionComponentValueModel):

    pass