from Mindblocks.model.execution_graph.execution_component_model import ExecutionComponentModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository
from Mindblocks.repository.execution_component_repository.execution_component_specifications import \
    ExecutionComponentSpecifications


class ExecutionComponentRepository(AbstractRepository):

    def __initialize_model__(self):
        return ExecutionComponentModel()

    def get_specifications(self):
        return ExecutionComponentSpecifications()

    def create(self, specifications):
        model = self.__create__()
        model.name = specifications.name
        model.mode = specifications.mode

        return model

    def create_from_creation_component(self, component):
        # TODO: This is a little unprincipled, but can be refactored later
        model = self.new()

        model.execution_type = component.component_type
        model.component_identifier = component.identifier
        model.language = component.language
        model.name = component.name

        return model