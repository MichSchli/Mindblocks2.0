class EdgeModel:

    source_socket = None
    target_socket = None

    def __init__(self, source_socket, target_socket):
        self.source_socket = source_socket
        self.target_socket = target_socket