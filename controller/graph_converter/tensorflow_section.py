class TensorflowSection:

    components = None
    outputs = None

    def __init__(self):
        self.components = []
        self.matched_in_sockets = []
        self.matched_out_sockets = []

    def add_component(self, component):
        self.components.append(component)

    def map_out_socket(self, out_socket, new_out_socket):
        self.matched_out_sockets.append((out_socket, new_out_socket))

    def map_in_socket(self, in_socket, new_in_socket):
        self.matched_in_sockets.append((in_socket, new_in_socket))

    def compile(self):
        self.fill_in_sockets()
        self.compile_graph()

    def compile_graph(self):
        self.outputs = [tf_out_socket.pull() for tf_out_socket, _ in self.matched_out_sockets]

    def fill_in_sockets(self):
        for tf_in_socket, in_socket in self.matched_in_sockets:
            socket_type = in_socket.pull_type()
            socket_dim = in_socket.pull_dim()

            socket_placeholder = self.get_placeholder(socket_type, socket_dim)
            tf_in_socket.set_cached_value(socket_placeholder)

    def execute(self):
        feed_dict = {}
        for tf_in_socket, in_socket in self.matched_in_sockets:
            placeholder = tf_in_socket.pull()
            value = in_socket.pull()
            feed_dict[placeholder] = value

        tf_outputs = self.session.run(self.outputs, feed_dict=feed_dict)

        for i in range(len(self.matched_out_sockets)):
            self.matched_out_sockets[i][1].set_cached_value(tf_outputs[i])