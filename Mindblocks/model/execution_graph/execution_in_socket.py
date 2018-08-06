import numpy as np


class ExecutionInSocket:

    source = None
    execution_component = None

    replaced_value = None
    cast = None

    def set_source(self, execution_out_socket):
        self.source = execution_out_socket

    def pull(self, mode):
        if self.replaced_value is not None:
            return self.replaced_value

        return self.source.pull(mode)

    def pull_type_model(self):
        source_type = self.source.pull_type_model()

        if self.cast is not None:
            return source_type.cast(self.cast)
        else:
            return source_type

    def clear_caches(self):
        if self.source is not None:
            self.source.clear_caches()

    def has_batches(self):
        return self.source.has_batches()

    def replace_value(self, value):
        self.replaced_value = value