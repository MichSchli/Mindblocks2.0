class TensorflowSection:

    components = None

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

    def execute(self):
        for tf_in_socket, in_socket in self.matched_in_sockets:
            tf_in_socket.set_cached_value(in_socket.pull())

        for tf_out_socket, execution_out_socket in self.matched_out_sockets:
            execution_out_socket.set_cached_value(tf_out_socket.pull())