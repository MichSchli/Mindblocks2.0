from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf


class LstmCell(ComponentTypeModel):

    name = "LstmCell"
    in_sockets = ["input_x", "previous_c", "previous_h"]
    out_sockets = ["output_c", "output_h"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return LstmCellValue()

    def build_value_type_model(self, in_types, execution_value):
        return {"output_c": in_types["previous_c"].copy(),
                "output_h": in_types["previous_h"].copy()}

    def execute(self, input_dictionary, execution_value, output_value_models, mode):
        input_x = input_dictionary["input_x"].get_value()
        previous_c = input_dictionary["previous_c"].get_value()
        previous_h = input_dictionary["previous_h"].get_value()

        cell_input = (previous_c, previous_h)

        new_h, new_state = execution_value.cell(input_x, cell_input)
        new_c = new_state[0]

        output_value_models["output_c"].assign(new_c)
        output_value_models["output_h"].assign(new_h)

        return output_value_models


class LstmCellValue(ExecutionComponentValueModel):

    def __init__(self):
        self.cell = tf.nn.rnn_cell.LSTMCell(20)