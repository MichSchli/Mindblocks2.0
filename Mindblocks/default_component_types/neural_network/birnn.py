import tensorflow as tf

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class BiRnn(ComponentTypeModel):

    name = "BiRnn"
    in_sockets = ["input"]
    out_sockets = ["output", "final_state", "layer_final_states"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        layers = 1
        if "layers" in value_dictionary:
            layers = int(value_dictionary["layers"][0][0])

        value = BiRnnValue(value_dictionary["dimension"][0][0], layers=layers)

        if "layer_dropout" in value_dictionary:
            value.set_layer_dropout(float(value_dictionary["layer_dropout"][0][0]))

        return value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        sequences = input_dictionary["input"].get_value()
        lengths = input_dictionary["input"].get_lengths()[1]

        layer_final_states = []

        for layer in range(value.layers):
            cell_forward = value.cells_forward[layer]
            cell_backward = value.cells_backward[layer]

            raw_out_sequences, final_states = tf.nn.bidirectional_dynamic_rnn(cell_forward,
                                                         cell_backward,
                                                         sequences,
                                                         dtype=tf.float32,
                                                         sequence_length=lengths,
                                                         scope=value.get_name())

            out_sequences = tf.concat(raw_out_sequences, axis=-1)
            sequences = out_sequences

            if mode == "train" and value.layer_dropout_keep_prob is not None:
                sequences = tf.nn.dropout(sequences, value.layer_dropout_keep_prob)

            final_states = tf.concat([final_states[i][1] for i in [0,1]], -1)
            layer_final_states.append(final_states)

        layer_final_states = tf.stack(layer_final_states, 1)

        output_value_models["output"].assign(out_sequences, length_list=[None, lengths, None])
        output_value_models["final_state"].assign(final_states, length_list=None)
        output_value_models["layer_final_states"].assign(layer_final_states, length_list=None)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        new_type = input_types["input"].copy()
        value.input_dimension = new_type.get_dimension(-1)

        output_hidden_dim = value.get_final_cell_size()
        new_type.set_dimension(-1, output_hidden_dim)

        final_state_type = SoftTensorTypeModel([new_type.get_dimension(0), output_hidden_dim], string_type="float")
        layer_final_state_type = SoftTensorTypeModel([new_type.get_dimension(0), value.layers, output_hidden_dim], string_type="float")
        return {"output": new_type,
                "final_state": final_state_type,
                "layer_final_states": layer_final_state_type}

class BiRnnValue(ExecutionComponentValueModel):

    layers = None
    layer_dropout_keep_prob = None

    def __init__(self, cell_size, layers=1):
        self.layers = layers
        self.cell_size = int(cell_size)

        self.cells_forward = [None] * layers
        self.cells_backward = [None] * layers

        for i in range(layers):
            self.cells_forward[i] = tf.nn.rnn_cell.LSTMCell(self.cell_size / 2, name= self.get_name() + "-forward_"+str(i))
            self.cells_backward[i] = tf.nn.rnn_cell.LSTMCell(self.cell_size / 2, name=self.get_name() + "-backward_"+str(i))

        self.cell_forward = tf.nn.rnn_cell.LSTMCell(self.cell_size / 2, name= self.get_name() + "-forward")
        self.cell_backward = tf.nn.rnn_cell.LSTMCell(self.cell_size / 2, name= self.get_name() + "-backward")

    def set_layer_dropout(self, dropout):
        self.layer_dropout_keep_prob = 1 - dropout

    def get_final_cell_size(self):
        return self.cell_size

    def count_parameters(self):
        parameters = 0
        direction_output_dim = int(self.cell_size / 2)

        input_dim = self.input_dimension
        output_dim = self.cell_size

        for layer in range(self.layers):
            if layer > 0:
                input_dim = output_dim

            parameters += 2 * (4 * direction_output_dim * (input_dim + direction_output_dim + 1))

        return parameters