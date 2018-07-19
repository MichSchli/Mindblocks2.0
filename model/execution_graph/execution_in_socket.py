class ExecutionInSocket:

    source = None

    def set_source(self, execution_out_socket):
        self.source = execution_out_socket

    def pull(self):
        return self.source.pull()