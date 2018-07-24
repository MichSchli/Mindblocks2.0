

class ExecutionInSocket:

    source = None
    execution_component = None

    replaced_value = None
    cast = None

    def set_source(self, execution_out_socket):
        self.source = execution_out_socket

    def pull(self):
        if self.replaced_value is not None:
            return self.replaced_value

        return self.source.pull()

    def pull_type(self):
        if self.cast is not None:
            return self.cast
        return self.source.pull_type()

    def pull_dim(self):
        return self.source.pull_dim()

    def clear_caches(self):
        self.source.clear_caches()