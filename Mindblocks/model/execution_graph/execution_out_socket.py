class ExecutionOutSocket:

    cached_value = None
    cached_type = None

    execution_component = None
    targets = None

    def __init__(self):
        self.targets = []

    def pull(self, mode):
        if self.cached_value is None:
            self.execution_component.execute(mode)

        return self.cached_value

    def set_cached_value(self, value):
        self.cached_value = value

    def add_target(self, in_socket):
        self.targets.append(in_socket)

    def clear_caches(self):
        self.cached_value = None
        self.execution_component.clear_caches()

    def has_batches(self):
        return self.execution_component.has_batches()

    def pull_type_model(self):
        if self.cached_type is None:
            self.execution_component.infer_type_models()
        return self.cached_type

    def set_cached_type(self, type_model):
        self.cached_type = type_model