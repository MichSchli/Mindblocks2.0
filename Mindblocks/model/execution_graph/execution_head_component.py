class ExecutionHeadComponent:

    run_output_sockets = None
    language = "python"

    def __init__(self):
        self.run_output_sockets = []

    def add_in_socket(self, in_socket):
        self.run_output_sockets.append(in_socket)

    def clear_caches(self):
        for socket in self.run_output_sockets:
            socket.clear_caches()

    def pull(self, mode):
        return [socket.pull(mode) for socket in self.run_output_sockets]

    def has_batches(self):
        for socket in self.run_output_sockets:
            if not socket.has_batches():
                return False
        return True

    def initialize_type_models(self):
        [socket.pull_type_model() for socket in self.run_output_sockets]