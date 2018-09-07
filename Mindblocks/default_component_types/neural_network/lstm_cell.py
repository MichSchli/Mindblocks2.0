from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class LstmCell(ComponentTypeModel):

    name = "LstmCell"
    in_sockets = ["input_x", "previous_c", "previous_h"]
    out_sockets = ["output_c", "output_h", "layer_output_cs", "layer_output_hs"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = LstmCellValue(int(value_dictionary["dimension"][0][0]))

        if "layers" in value_dictionary:
            value.set_layers(int(value_dictionary["layers"][0][0]))

        if "layer_dropout" in value_dictionary:
            value.set_layer_dropout(float(value_dictionary["layer_dropout"][0][0]))

        value.initialize_tensorflow_variable()

        return value

    def build_value_type_model(self, in_types, execution_value, mode):
        batch_size = in_types["previous_c"].get_batch_size()
        execution_value.input_dimension = in_types["input_x"].get_inner_dim()

        new_h_type = TensorTypeModel("float", [batch_size, execution_value.get_final_cell_size()])
        new_c_type = TensorTypeModel("float", [batch_size, execution_value.get_final_cell_size()])

        if execution_value.layers == 1:
            new_h_layers_type = TensorTypeModel("float", [batch_size, execution_value.get_final_cell_size()])
            new_c_layers_type = TensorTypeModel("float", [batch_size, execution_value.get_final_cell_size()])
        else:
            new_h_layers_type = TensorTypeModel("float", [batch_size, execution_value.layers, execution_value.get_final_cell_size()])
            new_c_layers_type = TensorTypeModel("float", [batch_size, execution_value.layers, execution_value.get_final_cell_size()])

        return {"output_c": new_c_type,
                "output_h": new_h_type,
                "layer_output_cs": new_c_layers_type,
                "layer_output_hs": new_h_layers_type}

    def execute(self, execution_component, input_dictionary, execution_value, output_value_models, mode):
        input_x = input_dictionary["input_x"].get_value()
        previous_c = input_dictionary["previous_c"].get_value()
        previous_h = input_dictionary["previous_h"].get_value()

        new_hs = []
        new_cs = []

        if execution_value.layers == 1:
            previous_c_shape = tf.shape(previous_c)
            previous_h_shape = tf.shape(previous_h)

            previous_c = tf.reshape(previous_c, [previous_c_shape[0], 1, previous_c_shape[-1]])
            previous_h = tf.reshape(previous_c, [previous_h_shape[0], 1, previous_h_shape[-1]])

        if False: #execution_value.layers == 1:
            cell_input = (previous_c, previous_h)

            h, new_state = execution_value.cells[0](input_x, cell_input)
            c = new_state[0]

            final_cs = c
            final_hs = h
        else:
            layer_input = input_x
            for i in range(execution_value.layers):
                cell_input = (previous_c[:,i,:], previous_h[:,i,:])
                h, new_state = execution_value.cells[i](layer_input, cell_input)
                c = new_state[0]

                new_cs.append(c)
                new_hs.append(h)

                layer_input = h

                if mode == "train" and execution_value.layer_dropout_keep_prob is not None:
                    layer_input = tf.nn.dropout(layer_input, execution_value.layer_dropout_keep_prob)

            final_cs = new_cs[-1]
            final_hs = new_hs[-1]

            new_cs = tf.stack(new_cs, axis=1)
            new_hs = tf.stack(new_hs, axis=1)

        output_value_models["output_c"].assign(final_cs)
        output_value_models["output_h"].assign(final_hs)

        output_value_models["layer_output_cs"].assign(new_cs)
        output_value_models["layer_output_hs"].assign(new_hs)

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
        self.cells = [tf.nn.rnn_cell.LSTMCell(self.dimension) for _ in range(self.layers)]

    def get_final_cell_size(self):
        return self.dimension

    def count_parameters(self):
        parameters = 0

        input_dim = self.input_dimension
        output_dim = self.dimension

        for layer in range(self.layers):
            if layer > 0:
                input_dim = output_dim

            parameters += 4 * output_dim * (input_dim + output_dim + 1)

        return parameters