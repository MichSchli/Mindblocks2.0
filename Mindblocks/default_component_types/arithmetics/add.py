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
        return AddValue()

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        result = input_dictionary["left"].get_value() + input_dictionary["right"].get_value()
        lengths = input_dictionary["left"].get_lengths()
        output_value_models["output"].assign(result, length_list=lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left_type = input_types["left"]
        right_type = input_types["right"]

        binary_op_helper = SoftTensorBinaryOperatorHelper()
        output_dimensions, soft_by_dimension = binary_op_helper.get_combine_type_dimensions(left_type, right_type, -1, value.get_name())

        # Get string data type:
        data_type = input_types["left"].get_data_type()

        output_type = SoftTensorTypeModel(output_dimensions,
                                          soft_by_dimensions=soft_by_dimension,
                                          string_type=data_type)

        return {"output": output_type}

class AddValue(ExecutionComponentValueModel):

    pass