class EdgeModel:

    source_socket = None
    target_socket = None

    def __init__(self, source_socket, target_socket):
        self.source_socket = source_socket
        self.target_socket = target_socket

    def get_source_component_name(self):
        return self.source_socket.get_component().name

    def get_target_component_name(self):
        return self.target_socket.get_component().name