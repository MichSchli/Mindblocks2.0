from Mindblocks.error_handling.connectivity.unconnected_socket_exception import UnconnectedSocketException
from Mindblocks.model.abstract.abstract_execution_model import AbstractExecutionModel


class ExecutionInSocket(AbstractExecutionModel):


    """
    New:
    """

    edge = None

    def add_edge(self, edge):
        self.edge = edge

    """
    Old:
    """

    execution_component = None

    replaced_value = None
    replaced_type = None

    def pull(self, mode):
        if self.replaced_value is not None:
            value = self.replaced_value
        else:
            value = self.edge.pull(mode)

        return value

    def get_name(self):
        #TODO
        return self.execution_component.get_name()

    def pull_type_model(self, mode):
        if self.replaced_type is not None:
            return self.replaced_type

        if self.edge is None:
            raise UnconnectedSocketException("Attempted type pull from unconnected in socket \"" + self.get_description() + "\"")

        source_type = self.edge.pull_type_model(mode)
        return source_type

    def clear_caches(self):
        if self.edge is not None:
            self.edge.clear_caches()

    def has_batches(self, mode):
        return self.edge.has_batches(mode)

    def replace_type(self, type):
        self.replaced_type = type

    def replace_value(self, value):
        self.replaced_value = value

    def describe_graph(self, indent=0):
        if self.edge is not None:
            self.edge.describe_graph(indent=indent)

    def init_batches(self):
        if self.edge is not None:
            self.edge.init_batches()

    def should_use_placeholder_for_tensorflow(self):
        return self.edge.source.should_use_placeholder_for_tensorflow()

    def initialize(self, mode, tensorflow_session_model):
        if self.edge is not None:
            return self.edge.initialize(mode, tensorflow_session_model)
        elif self.replaced_value is not None:
            return self.replaced_value

    def get_past(self):
        if self.edge is not None:
            return self.edge.get_past()
        else:
            return []