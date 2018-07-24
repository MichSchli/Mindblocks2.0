class ExecutionOutSocket:

    cached_value = None
    cached_type = None
    cached_dims = None

    execution_component = None
    targets = None

    def __init__(self):
        self.targets = []

    def pull(self):
        if self.cached_value is None:
            self.execution_component.execute()

        return self.cached_value

    def pull_type(self):
        if self.cached_type is None:
            self.execution_component.infer_types()
        return self.cached_type

    def pull_dim(self):
        if self.cached_dims is None:
            self.execution_component.infer_dims()
        return self.cached_dims

    def set_cached_value(self, value):
        self.cached_value = value

    def set_cached_dims(self, dims):
        self.cached_dims = dims

    def set_cached_type(self, type):
        self.cached_type = type

    def add_target(self, in_socket):
        self.targets.append(in_socket)

    def clear_caches(self):
        self.cached_value = None
        self.execution_component.clear_caches()