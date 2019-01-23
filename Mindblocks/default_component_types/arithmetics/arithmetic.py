from Mindblocks.helpers.soft_tensors.soft_tensor_binary_operator_helper import SoftTensorBinaryOperatorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class Arithmetic(ComponentTypeModel):

    name = "Arithmetic"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        operation = value_dictionary["operation"][0][0]
        return ArithmeticValue(operation)

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        left = input_dictionary["left"]
        right = input_dictionary["right"]

        if value.operation == "minus":
            op = tf.subtract
        elif value.operation == "add":
            op = tf.add
        elif value.operation == "mul":
            op = tf.multiply

        helper = SoftTensorBinaryOperatorHelper()
        helper.process(left, right, op, output_value_models["output"], language="tensorflow")

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left_type = input_types["left"]
        right_type = input_types["right"]

        helper = SoftTensorBinaryOperatorHelper()
        output_type = helper.create_output_type(left_type, right_type, left_type.get_data_type(), value.get_name())

        return {"output": output_type}

class ArithmeticValue(ExecutionComponentValueModel):

    def __init__(self, operation):
        self.operation = operation