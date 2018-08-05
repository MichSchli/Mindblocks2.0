import tensorflow as tf


class TensorflowSection:

    components = None
    outputs = None
    session = None
    execution_type = None

    def __init__(self):
        self.components = []
        self.matched_in_sockets = []
        self.matched_out_sockets = []

    def add_component(self, component):
        self.components.append(component)

    def set_session(self, session):
        self.session = session

    def map_out_socket(self, out_socket, new_out_socket):
        self.matched_out_sockets.append((out_socket, new_out_socket))

    def map_in_socket(self, in_socket, new_in_socket):
        self.matched_in_sockets.append((in_socket, new_in_socket))

    def compile(self, mode):
        self.outputs = [tf_out_socket.pull(mode) for tf_out_socket, _ in self.matched_out_sockets]

    def initialize_placeholders(self):
        for tf_in_socket, in_socket in self.matched_in_sockets:
            socket_placeholder = self.get_placeholder(in_socket)

            # TODO: OVERLAPPING IN SOCKETS (same out socket) USE SEPARATE PLACEHOLDERS
            tf_in_socket.replaced_value = socket_placeholder

    def get_placeholder(self, in_socket):
        value_type = in_socket.pull_value_type()
        return value_type.get_tensorflow_placeholder()

    def execute(self, mode):
        feed_dict = self.create_feed_dict(mode)
        tf_outputs = self.session.run(self.outputs, feed_dict=feed_dict)

        for i in range(len(self.matched_out_sockets)):
            output_type = self.matched_out_sockets[i][0].pull_value_type()
            formatted_output = output_type.format_from_tensorflow_output(tf_outputs[i])
            self.matched_out_sockets[i][1].set_cached_value(formatted_output)

    def create_feed_dict(self, mode):
        feed_dict = {}
        for tf_in_socket, in_socket in self.matched_in_sockets:
            placeholder = tf_in_socket.pull(mode)
            value = in_socket.pull(mode)
            value_type = in_socket.pull_value_type()

            feed_value = value_type.format_for_tensorflow_input(value)

            feed_dict[placeholder] = feed_value
        return feed_dict

    def infer_types(self):
        for tf_out_socket, out_socket in self.matched_out_sockets:
            out_type = tf_out_socket.pull_type()
            out_socket.set_cached_type(out_type)

    def infer_dims(self):
        for tf_out_socket, out_socket in self.matched_out_sockets:
            out_dims = tf_out_socket.pull_dim()
            out_socket.set_cached_dims(out_dims)

    def clear_caches(self):
        for _, in_socket in self.matched_in_sockets:
            in_socket.clear_caches()

    def has_batches(self):
        for _, in_socket in self.matched_in_sockets:
            if not in_socket.has_batches():
                return False
        return True
