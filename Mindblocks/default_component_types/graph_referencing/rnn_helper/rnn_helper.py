from tensorflow.python.ops import tensor_array_ops

from Mindblocks.default_component_types.graph_referencing.rnn_helper.rnn_model import RnnModel
import tensorflow as tf

from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class RnnHelper:

    def __init__(self):
        pass

    def tile_batches(self, rnn_model, tiling_factor):
        rnn_model.tiling_factor = tiling_factor

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
                parts = graph_input.split(":")
                in_socket = rnn_model.inner_graph.get_in_socket(parts[0], parts[1])
                value = in_socket.replaced_type.initialize_value_model("tensorflow")

                input_value = input_dictionary[component_input].get_value()
                input_lens = input_dictionary[component_input].get_lengths()

                tf_inp = tf.contrib.seq2seq.tile_batch(input_value, rnn_model.tiling_factor)

                tf_inp_lens = [tf.contrib.seq2seq.tile_batch(l, rnn_model.tiling_factor) if l is not None else None for l in input_lens]

                value.assign(tf_inp, length_list=tf_inp_lens)

                rnn_model.inner_graph.enforce_value(parts[0], parts[1], value)
            elif feed_type != "loop":
                rnn_model.inner_graph.enforce_value(parts[0], parts[1], input_dictionary[component_input])

    def add_sequence_outputs(self, rnn_model, maximum_iterations, mode):
        sequence_output_values = self.get_sequence_output_values(rnn_model,
                                                                 mode,
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

        batch_size = rnn_model.get_batch_size()

        for graph_output, graph_input, init in rnn_model.recurrences:
            if init is not None and init.startswith("zero_tensor"):
                counter += 1
                parts = graph_input.split(":")
                in_socket = rnn_model.inner_graph.get_in_socket(parts[0], parts[1])
                dims = in_socket.replaced_type.get_dimensions()
                dims[0] = batch_size * rnn_model.tiling_factor if batch_size is not None else None
                tf_type = tf.int32 if in_socket.replaced_type.get_data_type() == "int" else tf.float32
                tf_value = tf.zeros(dims, dtype=tf_type, name="zero_initializer_"+str(counter))

                initializers.append(tf_value)
                recurrency_sockets.append(in_socket)
            elif init is not None and init.startswith("socket"):
                parts = graph_input.split(":")
                in_socket = rnn_model.inner_graph.get_in_socket(parts[0], parts[1])
                linked_socket = input_dictionary[init[7:]]

                init_value = linked_socket.get_value()

                if rnn_model.tiling_factor > 1:
                    init_value = tf.contrib.seq2seq.tile_batch(init_value, rnn_model.tiling_factor)

                initializers.append(init_value)
                recurrency_sockets.append(in_socket)

        return recurrency_sockets, initializers

    def get_sequence_output_values(self, rnn_model, mode, maximum_iterations=None):
        sequence_output_values = []
        counter = 0
        for _, graph_output, desc in rnn_model.out_links:
            desc_parts = desc.split("|")
            if len(desc_parts) > 1 and desc_parts[1] == "int":
                tf_type = tf.int32
            else:
                tf_type = tf.float32

            counter += 1
            # use tensor arrays
            parts = graph_output.split(":")
            socket = rnn_model.inner_graph.get_out_socket(parts[0], parts[1])
            out_type = socket.pull_type_model(mode)
            dims = out_type.get_dimensions()

            tf_value = tensor_array_ops.TensorArray(
                dtype=tf_type,
                size=0 if maximum_iterations is None else maximum_iterations,
                dynamic_size=maximum_iterations is None,
                element_shape=dims,
                name="sequence_output_"+str(counter))
            sequence_output_values.append(tf_value)

        return sequence_output_values

    def handle_input_types(self, rnn_model, input_type_dictionary):
        batch_size = None
        for component_input, graph_input, feed_type in rnn_model.in_links:
            parts = graph_input.split(":")
            source_input_type = input_type_dictionary[component_input]

            if feed_type == "loop":
                graph_input_type = source_input_type.get_single_token_type()
            elif feed_type == "per_batch" or feed_type == "initializer":
                graph_input_type = source_input_type.copy()
            else:
                graph_input_type = source_input_type.copy()

            batch_inp_size = graph_input_type.get_dimension(0)
            if batch_inp_size is not None:
                batch_inp_size *= rnn_model.tiling_factor

                graph_input_type.set_dimension(0, batch_inp_size)

            rnn_model.inner_graph.enforce_type(parts[0], parts[1], graph_input_type)

        for graph_output, graph_input, init in rnn_model.recurrences:
            if init is not None and init.startswith("zero_tensor"):
                parts = graph_input.split(":")
                init_info = init[12:].split("|")
                init_type = init_info[1] if len(init_info) > 1 else "float"

                dims = [batch_size] + [int(v) for v in init_info[0].split(",")] if len(init_info[0]) > 0 else [
                    batch_size]
                tensor_type = SoftTensorTypeModel(dims, string_type=init_type)
                rnn_model.inner_graph.enforce_type(parts[0], parts[1], tensor_type)
            elif init is not None and init.startswith("socket:"):
                parts = graph_input.split(":")
                init_info = init.split(":")[1]
                source_input_type = input_type_dictionary[init_info]
                input_type = source_input_type.copy()
                rnn_model.inner_graph.enforce_type(parts[0], parts[1], input_type)