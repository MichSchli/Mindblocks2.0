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
        left = input_dictionary["left"].get_value()
        right = input_dictionary["right"].get_value()

        left_dims = len(left.shape)
        right_dims = len(right.shape)

        if left_dims > right_dims:
            out_lengths = input_dictionary["left"].get_lengths()
        else:
            out_lengths = input_dictionary["right"].get_lengths()

        for dim in range(left_dims, right_dims):
            left = tf.expand_dims(left, -1)

        for dim in range(right_dims, left_dims):
            right = tf.expand_dims(right, -1)

        if value.operation == "minus":
            result = tf.subtract(left, right)
        elif value.operation == "add":
            result = tf.add(left, right)
        elif value.operation == "mul":
            result = tf.multiply(left, right)

        output_value_models["output"].assign(result, length_list=out_lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        left_dims = len(input_types["left"].get_dimensions())
        right_dims = len(input_types["right"].get_dimensions())

        # TODO: this is a hack
        if left_dims > right_dims:
            return {"output": input_types["left"].copy()}
        else:
            return {"output": input_types["right"].copy()}

class ArithmeticValue(ExecutionComponentValueModel):

    def __init__(self, operation):
        self.operation = operation