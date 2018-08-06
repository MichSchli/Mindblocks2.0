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
        self.outputs = [tf_out_socket.pull(mode).get_tensorflow_output_tensors() for tf_out_socket, _ in self.matched_out_sockets]

    def initialize_placeholders(self):
        for tf_in_socket, in_socket in self.matched_in_sockets:
            socket_placeholder = self.get_placeholder(in_socket)

            tf_in_socket.replaced_value = socket_placeholder

    def get_placeholder(self, in_socket):
        value_type = in_socket.pull_type_model()
        return value_type.get_tensorflow_placeholder()

    def execute(self, mode):
        feed_dict = self.create_feed_dict(mode)
        tf_outputs = self.session.run(self.outputs, feed_dict=feed_dict)

        for i in range(len(self.matched_out_sockets)):
            output_type = self.matched_out_sockets[i][0].pull_type_model()
            formatted_output = output_type.format_from_tensorflow_output(tf_outputs[i])
            self.matched_out_sockets[i][1].set_cached_value(formatted_output)

    def create_feed_dict(self, mode):
        feed_dict = {}
        for tf_in_socket, in_socket in self.matched_in_sockets:
            value = in_socket.pull(mode)
            placeholders = in_socket.pull_type_model().get_cached_placeholders()

            feed_values = value.format_for_tensorflow_input()
            for k,v in zip(placeholders, feed_values):
                feed_dict[k] = v

        return feed_dict

    def clear_caches(self):
        for _, in_socket in self.matched_in_sockets:
            in_socket.clear_caches()

    def has_batches(self):
        for _, in_socket in self.matched_in_sockets:
            if not in_socket.has_batches():
                return False
        return True
