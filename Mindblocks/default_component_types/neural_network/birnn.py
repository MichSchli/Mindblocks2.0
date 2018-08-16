from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf


class BiRnn(ComponentTypeModel):

    name = "BiRnn"
    in_sockets = ["input"]
    out_sockets = ["output", "final_state"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return BiRnnValue(value_dictionary["dimension"][0][0])

    def execute(self, input_dictionary, value, output_value_models, mode):
        cell_forward = value.cell_forward
        cell_backward = value.cell_backward

        sequences = input_dictionary["input"].get_sequences()
        lengths = input_dictionary["input"].get_sequence_lengths()

        rnn_output = tf.nn.bidirectional_dynamic_rnn(cell_forward,
                                                     cell_backward,
                                                     sequences,
                                                     dtype=tf.float32,
                                                     sequence_length=lengths)


        out_sequences = tf.concat(rnn_output[0], -1)
        final_states = tf.concat([rnn_output[1][i][1] for i in [0,1]], -1)

        output_value_models["output"].assign(out_sequences, language="tensorflow")
        output_value_models["final_state"].assign(final_states)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        new_type = input_types["input"].copy()
        new_type.set_inner_dim(value.get_final_cell_size())

        final_state_type = input_types["input"].get_single_token_type()
        final_state_type.extend_outer_dim(input_types["input"].get_batch_size())
        return {"output": new_type,
                "final_state": final_state_type}

class BiRnnValue(ExecutionComponentValueModel):

    def __init__(self, cell_size):
        self.cell_size = int(cell_size)
        self.cell_forward = tf.nn.rnn_cell.LSTMCell(self.cell_size, num_proj=self.cell_size/2, name="forward")
        self.cell_backward = tf.nn.rnn_cell.LSTMCell(self.cell_size, num_proj=self.cell_size/2, name="backward")

    def get_final_cell_size(self):
        return self.cell_size