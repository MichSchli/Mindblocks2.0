import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_model import ExecutionComponentModel


class TensorflowSection(ExecutionComponentModel):

    components = None
    outputs = None
    session_model = None
    execution_type = None

    def __init__(self):
        self.components = []
        self.matched_in_sockets = []
        self.matched_out_sockets = []

    def add_component(self, component):
        self.components.append(component)

    def set_session(self, session):
        self.session_model = session

    def map_out_socket(self, out_socket, new_out_socket):
        self.matched_out_sockets.append((out_socket, new_out_socket))

    def map_in_socket(self, in_socket, new_in_socket):
        self.matched_in_sockets.append((in_socket, new_in_socket))

    def get_in_sockets(self):
        return [m[1] for m in self.matched_in_sockets]

    def compile(self, mode):
        self.outputs = [tf_out_socket.pull(mode).get_tensorflow_output_tensors() for tf_out_socket, _ in self.matched_out_sockets]

    def initialize_placeholders(self, mode):
        for tf_in_socket, in_socket in self.matched_in_sockets:
            socket_placeholder = self.get_placeholder(in_socket, mode)

            tf_in_socket.replaced_value = socket_placeholder

    def get_placeholder(self, in_socket, mode):
        value_type = in_socket.pull_type_model(mode)
        return value_type.get_tensorflow_placeholder()

    def get_value(self):
        return self

    def count_parameters(self):
        parameters = 0

        for component in self.components:
            parameters += component.count_parameters()

        return parameters

    def execute(self, mode):
        feed_dict = self.create_feed_dict(mode)
        tf_outputs = self.session_model.run(self.outputs, feed_dict)

        for i in range(len(self.matched_out_sockets)):
            output_type = self.matched_out_sockets[i][0].pull_type_model(mode)
            formatted_output = output_type.format_from_tensorflow_output(tf_outputs[i])
            self.matched_out_sockets[i][1].set_cached_value(formatted_output)

    def create_feed_dict(self, mode):
        feed_dict = {}
        for tf_in_socket, in_socket in self.matched_in_sockets:
            if not self.should_use_placeholder(in_socket):
                continue

            value = in_socket.pull(mode)
            placeholders = in_socket.pull_type_model(mode).get_cached_placeholders()

            feed_values = value.format_for_tensorflow_input()
            for k,v in zip(placeholders, feed_values):
                feed_dict[k] = v

        return feed_dict

    def should_use_placeholder(self, in_socket):
        return in_socket.should_use_placeholder_for_tensorflow()

    def initialize(self, mode, tensorflow_session_model):
        for tf_in_socket, in_socket in self.matched_in_sockets:
            initialization_value = tf_in_socket.initialize(mode, tensorflow_session_model)

            if self.should_use_placeholder(in_socket):
                socket_placeholder = self.get_placeholder(in_socket, mode)
                tf_in_socket.replaced_value = socket_placeholder
            else:
                tf_in_socket.replaced_value = initialization_value

        for tf_out_socket, out_socket in self.matched_out_sockets:
            init_value = tf_out_socket.initialize(mode, tensorflow_session_model)
            out_socket.set_cached_init_value(init_value)

        self.compile(mode)

    def clear_caches(self):
        for _, in_socket in self.matched_in_sockets:
            in_socket.clear_caches()

    def describe_graph(self, indent=0):
        print("\t"*indent + "TF Section")

        for _, in_socket in self.matched_in_sockets:
            in_socket.describe_graph(indent=indent+1)

    def has_batches(self, mode):
        for _, in_socket in self.matched_in_sockets:
            if not in_socket.has_batches(mode):
                return False
        return True

    def get_referenced_components(self):
        cs = self.components[:]
        for referenced_graph in self.get_referenced_graphs():
            cs.extend(referenced_graph.get_components())
        return cs

    def init_batches(self):
        #TODO: We are not initing in graph
        for in_socket in self.get_in_sockets():
            in_socket.init_batches()

    def get_referenced_graphs(self):
        gs = []

        for component in self.components:
            gs.extend(component.get_referenced_graphs())

        return list(set(gs))