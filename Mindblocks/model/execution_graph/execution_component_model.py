class ExecutionComponentModel:

    out_sockets = None
    in_sockets = None
    identifier = None
    language = None

    def __init__(self):
        self.out_sockets = {}
        self.in_sockets = {}

    def add_out_socket(self, key, socket):
        self.out_sockets[key] = socket

    def add_in_socket(self, key, socket):
        self.in_sockets[key] = socket

    def execute(self, mode):
        input_dictionary = {k : in_socket.pull(mode) for k,in_socket in self.in_sockets.items()}
        output_dictionary = self.execution_type.execute(input_dictionary, self.execution_value, mode)

        for k,v in output_dictionary.items():
            self.out_sockets[k].set_cached_value(v)

    def infer_value_types(self):
        in_types = {k : in_socket.pull_value_type() for k,in_socket in self.in_sockets.items()}
        output_types = self.execution_type.build_value_type(in_types, self.execution_value)

        for k,v in output_types.items():
            self.out_sockets[k].set_cached_type(v)

    def infer_dims(self):
        in_dims = {k : in_socket.pull_dim() for k,in_socket in self.in_sockets.items()}
        output_dims = self.execution_type.infer_dims(in_dims, self.execution_value)

        for k,v in output_dims.items():
            self.out_sockets[k].set_cached_dims(v)

    def get_name(self):
        return self.name

    def get_in_sockets(self):
        return list(self.in_sockets.values())

    def get_out_sockets(self):
        return list(self.out_sockets.values())

    def clear_caches(self):
        for in_socket in self.get_in_sockets():
            in_socket.clear_caches()

    def has_batches(self):
        for in_socket in self.get_in_sockets():
            if not in_socket.has_batches():
                return False

        return self.execution_type.has_batches(self.execution_value)