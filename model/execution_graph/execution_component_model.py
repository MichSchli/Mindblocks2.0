class ExecutionComponentModel:

    out_sockets = None
    in_sockets = None

    def __init__(self):
        self.out_sockets = {}
        self.in_sockets = {}

    def add_out_socket(self, key, socket):
        self.out_sockets[key] = socket

    def add_in_socket(self, key, socket):
        self.in_sockets[key] = socket

    def execute(self):
        input_dictionary = {k : in_socket.pull() for k,in_socket in self.in_sockets.items()}
        output_dictionary = self.execution_type.execute(input_dictionary, self.execution_value)

        for k,v in output_dictionary.items():
            self.out_sockets[k].set_cached_value(v)