from Mindblocks.helpers.soft_tensors.soft_tensor_binary_operator_helper import SoftTensorBinaryOperatorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class Add(ComponentTypeModel):

    name = "Add"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = AddValue()
        value.language = language
        return value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        left = input_dictionary["left"]
        right = input_dictionary["right"]

        op = lambda x,y: x + y

        helper = SoftTensorBinaryOperatorHelper()
        helper.process(left, right, op, output_value_models["output"], language=value.language)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left_type = input_types["left"]
        right_type = input_types["right"]

        helper = SoftTensorBinaryOperatorHelper()
        output_type = helper.create_output_type(left_type, right_type, left_type.get_data_type(), value.get_name())

        return {"output": output_type}

class AddValue(ExecutionComponentValueModel):

    pass