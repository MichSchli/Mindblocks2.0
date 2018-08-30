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

    def init_batches(self):
        for socket in self.run_output_sockets:
            socket.init_batches()

    def initialize_type_models(self, mode):
        return [socket.pull_type_model(mode) for socket in self.run_output_sockets]

    def describe_graph(self, indent=0):
        print("\t"*indent + "Graph: ")
        for socket in self.run_output_sockets:
            socket.describe_graph(indent=indent+1)

    def initialize(self, mode, tensorflow_session_model):
        for socket in self.run_output_sockets:
            socket.initialize(mode, tensorflow_session_model)