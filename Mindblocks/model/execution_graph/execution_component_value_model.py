from Mindblocks.model.abstract.abstract_model import AbstractModel
from Mindblocks.model.abstract.abstract_execution_model import AbstractExecutionModel


class ExecutionComponentValueModel(AbstractModel):

    component_name = None
    component_mode = None

    def get_populate_items(self):
        return []

    def init_batches(self):
        self.has_batch = True

    def set_component_name(self, name, mode):
        self.component_name = name
        self.component_mode = mode

    def get_name(self):
        return str(self.component_name) + "-" + str(self.component_mode)

    def initialize_tensorflow_variables(self, tensorflow_session_model):
        pass

    def count_parameters(self):
        return 0

    def get_referenced_graphs(self):
        return []

    def get_description(self):
        return str(self.component_name) + ": value id " + str(self)