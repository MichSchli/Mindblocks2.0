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
        output_value_models = {k : type_model.initialize_value_model() for k,type_model in self.output_type_models.items()}

        print(self.execution_type.name)
        output_dictionary = self.execution_type.execute(input_dictionary, self.execution_value, output_value_models, mode)

        for k,v in output_dictionary.items():
            self.out_sockets[k].set_cached_value(v)

    def infer_type_models(self):
        in_types = {k : in_socket.pull_type_model() for k,in_socket in self.in_sockets.items()}
        self.output_type_models = self.execution_type.build_value_type_model(in_types, self.execution_value)


        for k,v in self.output_type_models.items():
            self.out_sockets[k].set_cached_type(v)

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