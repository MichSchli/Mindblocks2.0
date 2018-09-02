import numpy as np
import tensorflow as tf

class ExecutionInSocket:

    source = None
    execution_component = None

    replaced_value = None
    replaced_type = None
    cast = None
    dropout_rate = None

    def set_source(self, execution_out_socket):
        self.source = execution_out_socket

    def pull(self, mode):
        if self.replaced_value is not None:
            value = self.replaced_value
        else:
            value = self.source.pull(mode)

        if self.dropout_rate is not None and mode == "train":
            value.apply_dropouts(self.dropout_rate)

        if self.cast is not None:
            value.cast(self.cast)

        return value

    def pull_type_model(self, mode):
        if self.replaced_type is not None:
            return self.replaced_type

        source_type = self.source.pull_type_model(mode)

        if self.cast is not None:
            return source_type.cast(self.cast)
        else:
            return source_type

    def clear_caches(self):
        if self.source is not None:
            self.source.clear_caches()

    def has_batches(self):
        return self.source.has_batches()

    def replace_type(self, type):
        self.replaced_type = type

    def replace_value(self, value):
        self.replaced_value = value

    def describe_graph(self, indent=0):
        if self.source is not None:
            self.source.describe_graph(indent=indent)

    def init_batches(self):
        if self.source is not None:
            self.source.init_batches()

    def should_use_placeholder_for_tensorflow(self):
        return self.source.should_use_placeholder_for_tensorflow()

    def initialize(self, mode, tensorflow_session_model):
        if self.source is not None:
            return self.source.initialize(mode, tensorflow_session_model)
        elif self.replaced_value is not None:
            return self.replaced_value