class ExecutionHeadComponent:

    run_output_sockets = None
    language = "python"

    def __init__(self):
        self.run_output_sockets = []

    def add_in_socket(self, in_socket):
        self.run_output_sockets.append(in_socket)

    def pull(self):
        return [socket.pull() for socket in self.run_output_sockets]