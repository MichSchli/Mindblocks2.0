from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf


class BiRnn(ComponentTypeModel):

    name = "BiRnn"
    in_sockets = ["input"]
    out_sockets = ["output", "final_state"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        layers = 1
        if "layers" in value_dictionary:
            layers = int(value_dictionary["layers"][0][0])

        value = BiRnnValue(value_dictionary["dimension"][0][0], layers=layers)

        if "layer_dropout" in value_dictionary:
            value.set_layer_dropout(float(value_dictionary["layer_dropout"][0][0]))

        return value

    def execute(self, input_dictionary, value, output_value_models, mode):
        sequences = input_dictionary["input"].get_sequences()
        lengths = input_dictionary["input"].get_sequence_lengths()

        for layer in range(value.layers):
            cell_forward = value.cells_forward[layer]
            cell_backward = value.cells_backward[layer]

            rnn_output = tf.nn.bidirectional_dynamic_rnn(cell_forward,
                                                         cell_backward,
                                                         sequences,
                                                         dtype=tf.float32,
                                                         sequence_length=lengths)

            out_sequences = tf.concat(rnn_output[0], axis=-1)
            sequences = out_sequences

            if mode == "train" and value.layer_dropout_keep_prob is not None:
                sequences = tf.nn.dropout(sequences, value.layer_dropout_keep_prob)

        final_states = tf.concat([rnn_output[1][i][1] for i in [0,1]], -1)

        output_value_models["output"].assign_with_lengths(out_sequences, lengths, language="tensorflow")
        output_value_models["final_state"].assign(final_states)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        new_type = input_types["input"].copy()
        value.input_dimension = new_type.get_inner_dim()
        new_type.set_inner_dim(value.get_final_cell_size())

        final_state_type = new_type.get_single_token_type()
        final_state_type.extend_outer_dim(input_types["input"].get_batch_size())
        return {"output": new_type,
                "final_state": final_state_type}

class BiRnnValue(ExecutionComponentValueModel):

    layers = None
    layer_dropout_keep_prob = None

    def __init__(self, cell_size, layers=1):
        self.layers = layers
        self.cell_size = int(cell_size)

        self.cells_forward = [None] * layers
        self.cells_backward = [None] * layers

        for i in range(layers):
            self.cells_forward[i] = tf.nn.rnn_cell.LSTMCell(self.cell_size, num_proj=self.cell_size/2, name= self.get_name() + "-forward_"+str(i))
            self.cells_backward[i] = tf.nn.rnn_cell.LSTMCell(self.cell_size, num_proj=self.cell_size/2, name=self.get_name() + "-backward_"+str(i))

        self.cell_forward = tf.nn.rnn_cell.LSTMCell(self.cell_size, num_proj=self.cell_size/2, name= self.get_name() + "-forward")
        self.cell_backward = tf.nn.rnn_cell.LSTMCell(self.cell_size, num_proj=self.cell_size/2, name= self.get_name() + "-backward")

    def set_layer_dropout(self, dropout):
        self.layer_dropout_keep_prob = 1 - dropout

    def get_final_cell_size(self):
        return self.cell_size

    def count_parameters(self):
        parameters = 0

        input_dim = self.input_dimension
        output_dim = self.cell_size

        for layer in range(self.layers):
            if layer > 0:
                input_dim = output_dim

            parameters += 4 * output_dim * (input_dim + output_dim + 1)

        return parameters