class ExecutionOutSocket:

    cached_value = None
    execution_component = None

    def pull(self):
        if self.cached_value is None:
            self.execution_component.execute()

        return self.cached_value

    def set_cached_value(self, value):
        self.cached_value = value