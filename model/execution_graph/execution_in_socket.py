class ExecutionInSocket:

    source = None
    execution_component = None

    replaced_value = None

    def set_source(self, execution_out_socket):
        self.source = execution_out_socket

    def pull(self):
        if self.replaced_value is not None:
            return self.replaced_value

        return self.source.pull()

    def pull_type(self):
        return self.source.pull_type()

    def pull_dim(self):
        return self.source.pull_dim()