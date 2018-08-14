from tensorflow.python.ops import tensor_array_ops

from Mindblocks.default_component_types.graph_referencing.rnn_helper.rnn_model import RnnModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import tensorflow as tf


class RnnHelper:

    def __init__(self):
        pass

    def tile_batches(self, rnn_model, tiling_factor):
        rnn_model.tiling_factor = 3

    def create_rnn_model(self, value_dictionary):
        graph_name = value_dictionary["graph"][0][0]
        rnn_model = RnnModel(graph_name)

        if "in_link" in value_dictionary:
            for in_link in value_dictionary["in_link"]:
                parts = in_link[0].split("->")
                feed_type = in_link[1]["feed"] if "feed" in in_link[1] else None
                rnn_model.add_in_link(parts[0], parts[1], feed_type=feed_type)

        if "out_link" in value_dictionary:
            for out_link in value_dictionary["out_link"]:
                parts = out_link[0].split("->")
                feed_type = out_link[1]["feed"] if "feed" in out_link[1] else None
                rnn_model.add_out_link(parts[1], parts[0], feed_type=feed_type)

        if "recurrence" in value_dictionary:
            for recurrence in value_dictionary["recurrence"]:
                parts = recurrence[0].split("->")
                init = recurrence[1]["init"] if "init" in recurrence[1] else None
                rnn_model.add_recurrence(parts[0], parts[1], init=init)

        if "batch_size" in value_dictionary:
            rnn_model.batch_size = int(value_dictionary["batch_size"][0][0])

        return rnn_model

    def assign_static_inputs(self, rnn_model, input_dictionary):
        for component_input, graph_input, feed_type in rnn_model.in_links:
            parts = graph_input.split(":")
            if feed_type == "per_batch" and rnn_model.tiling_factor > 1:
                input_value = input_dictionary[component_input].get_value()
                parts = graph_input.split(":")
                in_socket = rnn_model.inner_graph.get_in_socket(parts[0], parts[1])
                value = in_socket.replaced_type.initialize_value_model()

                tf_inp = tf.contrib.seq2seq.tile_batch(input_value, rnn_model.tiling_factor)
                value.assign(tf_inp, language="tensorflow")

                rnn_model.inner_graph.enforce_value(parts[0], parts[1], value)
            elif feed_type != "loop":
                rnn_model.inner_graph.enforce_value(parts[0], parts[1], input_dictionary[component_input])

    def add_sequence_outputs(self, rnn_model, maximum_iterations):
        sequence_output_values = self.get_sequence_output_values(rnn_model,
                                                                 maximum_iterations=maximum_iterations)
        for sequence_output_value in sequence_output_values:
            rnn_model.add_loop_var(sequence_output_value)

    def add_recurrency_initializers(self, rnn_model, input_dictionary):
        in_sockets, initializer_values = self.get_recurrency_initializers(rnn_model, input_dictionary)
        for initializer_value in initializer_values:
            rnn_model.add_loop_var(initializer_value)
        rnn_model.list_of_in_sockets = in_sockets

    def get_recurrency_initializers(self, rnn_model, input_dictionary):
        recurrency_sockets = []
        initializers = []

        counter = 0

        for graph_output, graph_input, init in rnn_model.recurrences:
            if init is not None and init.startswith("zero_tensor"):
                counter += 1
                parts = graph_input.split(":")
                in_socket = rnn_model.inner_graph.get_in_socket(parts[0], parts[1])
                dims = in_socket.replaced_type.get_dimensions()
                tf_type = tf.int32 if in_socket.replaced_type.type == "int" else tf.float32
                tf_value = tf.zeros(dims, dtype=tf_type, name="zero_initializer_"+str(counter))
                initializers.append(tf_value)
                recurrency_sockets.append(in_socket)
            elif init is not None and init.startswith("socket"):
                parts = graph_input.split(":")
                in_socket = rnn_model.inner_graph.get_in_socket(parts[0], parts[1])
                linked_socket = input_dictionary[init[7:]]

                initializers.append(linked_socket.get_value())
                recurrency_sockets.append(in_socket)

        return recurrency_sockets, initializers

    def get_sequence_output_values(self, rnn_model, maximum_iterations=None):
        sequence_output_values = []
        counter = 0
        for _, graph_output, _ in rnn_model.out_links:
            counter += 1
            # use tensor arrays
            parts = graph_output.split(":")
            socket = rnn_model.inner_graph.get_out_socket(parts[0], parts[1])
            out_type = socket.pull_type_model()
            dims = out_type.get_dimensions()
            tf_value = tensor_array_ops.TensorArray(
                dtype=tf.float32,
                size=0 if maximum_iterations is None else maximum_iterations,
                dynamic_size=maximum_iterations is None,
                element_shape=dims,
                name="sequence_output_"+str(counter))
            sequence_output_values.append(tf_value)

        return sequence_output_values

    def handle_input_types(self, rnn_model, input_type_dictionary):
        batch_size = rnn_model.get_batch_size()

        if batch_size is None:
            for component_input, graph_input, feed_type in rnn_model.in_links:
                if feed_type == "per_batch":
                    source_input_type = input_type_dictionary[component_input]
                    batch_size = source_input_type.get_batch_size()

                    rnn_model.set_batch_size(batch_size)

        batch_size *= rnn_model.tiling_factor

        for component_input, graph_input, feed_type in rnn_model.in_links:
            parts = graph_input.split(":")
            source_input_type = input_type_dictionary[component_input]

            if feed_type == "loop":
                graph_input_type = source_input_type.get_single_token_type()
            elif feed_type == "per_batch" or feed_type == "initializer":
                graph_input_type = source_input_type.copy()
                graph_input_type.set_outer_dim(batch_size)
            else:
                graph_input_type = source_input_type

            rnn_model.inner_graph.enforce_type(parts[0], parts[1], graph_input_type)

        for graph_output, graph_input, init in rnn_model.recurrences:
            if init is not None and init.startswith("zero_tensor"):
                parts = graph_input.split(":")
                init_info = init[12:].split("|")
                init_type = init_info[1] if len(init_info) > 1 else "float"

                dims = [batch_size] + [int(v) for v in init_info[0].split(",")] if len(init_info[0]) > 0 else [
                    batch_size]
                tensor_type = TensorTypeModel(init_type, dims)
                rnn_model.inner_graph.enforce_type(parts[0], parts[1], tensor_type)
            elif init is not None and init.startswith("socket:"):
                parts = graph_input.split(":")
                input_type = graph_input_type.copy()
                input_type.set_outer_dim(batch_size)
                rnn_model.inner_graph.enforce_type(parts[0], parts[1], input_type)