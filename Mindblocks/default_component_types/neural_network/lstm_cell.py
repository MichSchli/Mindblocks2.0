from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf


class LstmCell(ComponentTypeModel):

    name = "LstmCell"
    in_sockets = ["input_x", "previous_c", "previous_h"]
    out_sockets = ["output_c", "output_h"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = LstmCellValue(int(value_dictionary["dimension"][0][0]))

        if "layers" in value_dictionary:
            value.set_layers(int(value_dictionary["layers"][0][0]))

        if "layer_dropout" in value_dictionary:
            value.set_layer_dropout(float(value_dictionary["layer_dropout"][0][0]))

        value.initialize_tensorflow_variable()

        return value

    def build_value_type_model(self, in_types, execution_value):
        return {"output_c": in_types["previous_c"].copy(),
                "output_h": in_types["previous_h"].copy()}

    def execute(self, input_dictionary, execution_value, output_value_models, mode):
        input_x = input_dictionary["input_x"].get_value()
        previous_c = input_dictionary["previous_c"].get_value()
        previous_h = input_dictionary["previous_h"].get_value()

        new_hs = []
        new_cs = []

        if execution_value.layers == 1:
            cell_input = (previous_c, previous_h)

            h, new_state = execution_value.cells[0](input_x, cell_input)
            c = new_state[0]

            new_hs = h
            new_cs = c
        else:
            for i in range(execution_value.layers):
                cell_input = (previous_c[i], previous_h[i])
                h, new_state = execution_value.cells[i](input_x, cell_input)
                c = new_state[0]

                new_cs.append(c)
                new_hs.append(h)

            new_cs = tf.concat(new_cs, 1)
            new_hs = tf.concat(new_hs, 1)

        output_value_models["output_c"].assign(new_cs)
        output_value_models["output_h"].assign(new_hs)

        return output_value_models


class LstmCellValue(ExecutionComponentValueModel):

    dimension = None
    layers = None
    layer_dropout_keep_prob = None

    cells = None

    def __init__(self, dimension):
        self.dimension = dimension
        self.layers = 1
        self.cells = []

    def set_layers(self, layers):
        self.layers = layers

    def set_layer_dropout(self, dropout_rate):
        self.layer_dropout_keep_prob = dropout_rate

    def initialize_tensorflow_variable(self):
        if self.layers == 1:
            self.cells = [tf.nn.rnn_cell.LSTMCell(self.dimension)]