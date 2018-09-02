import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops


class RnnModel:

    in_links = None
    out_links = None
    recurrences = None
    inner_graph = None
    inner_graph_name = None
    batch_size = None
    loop_vars = None

    tiling_factor = None

    def __init__(self, inner_graph_name):
        self.in_links = []
        self.out_links = []
        self.recurrences = []
        self.inner_graph_name = inner_graph_name
        self.loop_vars = []

        self.tiling_factor = 1

    def add_counter_loop_var(self):
        self.loop_vars.append(0)

    def add_length_var(self):
        length_var = tf.zeros(self.batch_size * self.tiling_factor, dtype=tf.int32)
        self.loop_vars.append(length_var)

    def add_finished_var(self):
        finished_var = tf.zeros(self.batch_size * self.tiling_factor, dtype=tf.bool)
        self.loop_vars.append(finished_var)

    def add_loop_var(self, var):
        self.loop_vars.append(var)

    def add_in_link(self, component_input, graph_input, feed_type=None):
        self.in_links.append((component_input, graph_input, feed_type))

    def add_out_link(self, component_output, graph_output, feed_type=None):
        self.out_links.append((component_output, graph_output, feed_type))

    def add_recurrence(self, graph_output, graph_input, init):
        self.recurrences.append((graph_output, graph_input, init))

    def set_inner_graph(self, graph):
        self.inner_graph = graph

    def get_graph_name(self):
        return self.inner_graph_name

    def get_required_graph_outputs(self):
        return [(l[0].split(":")[0], l[0].split(":")[1]) for l in self.recurrences] +\
               [(l[1].split(":")[0], l[1].split(":")[1]) for l in self.out_links]

    def get_required_graph_inputs(self):
        return [(l[1].split(":")[0], l[1].split(":")[1]) for l in self.in_links]

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def count_recurrent_links(self):
        return len(self.recurrences)

    def count_output_links(self):
        return len(self.out_links)

    def build_loop_var(self, description, name="var", extend_to_batch=True):
        parts = description.split("|")
        tf_type = tf.int32 if parts[1] == "int" else tf.float32

        dim_string = parts[0].split(":")[1]
        dims = [int(v) for v in dim_string.split(",")] if len(dim_string) > 0 else []

        if extend_to_batch:
            dims = [self.batch_size * self.tiling_factor] + dims

        tf_value = tf.zeros(dims, dtype=tf_type, name=name)

        self.add_loop_var(tf_value)

    def build_tensor_array_loop_var(self, description, name="var", extend_to_batch=True, maximum_iterations=None):
        parts = description.split("|")
        tf_type = tf.int32 if parts[1] == "int" else tf.float32

        dim_string = parts[0].split(":")[1]
        dims = [int(v) for v in dim_string.split(",")] if len(dim_string) > 0 else []

        if extend_to_batch:
            #dims = [self.batch_size * self.tiling_factor] + dims
            dims = [None] + dims

        tf_value = tensor_array_ops.TensorArray(
            dtype=tf_type,
            size=0 if maximum_iterations is None else maximum_iterations,
            dynamic_size=maximum_iterations is None,
            element_shape=dims,
            name=name)

        self.add_loop_var(tf_value)

    def set_nths_input(self, n, value):
        in_socket = self.list_of_in_sockets[n]
        type = in_socket.replaced_type
        value_model = type.initialize_value_model()
        value_model.assign(value)
        in_socket.replace_value(value_model)

    def get_inner_graph_output_types(self, mode):
        results = self.inner_graph.initialize_type_models(mode)
        out_type_dict = {}
        for output, result in zip(self.out_links, results):
            component_output, _, feed_type = output

            if feed_type.startswith("loop"):
                out_type = result.to_sequence_type()
            else:
                out_type = result

            out_type_dict[component_output] = out_type
        return out_type_dict

    def run(self):
        results = self.inner_graph.execute(discard_value_models=True)
        return results