class ExecutionOutSocket:

    cached_value = None
    execution_component = None
    targets = None

    def __init__(self):
        self.targets = []

    def pull(self):
        if self.cached_value is None:
            self.execution_component.execute()

        return self.cached_value

    def set_cached_value(self, value):
        self.cached_value = value

    def add_target(self, in_socket):
        self.targets.append(in_socket)