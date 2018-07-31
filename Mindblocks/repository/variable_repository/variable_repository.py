from Mindblocks.model.variable.variable_model import VariableModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository
from Mindblocks.repository.variable_repository.variable_specifications import VariableSpecifications


class VariableRepository(AbstractRepository):

    def __initialize_model__(self):
        return VariableModel()

    def get_specifications(self):
        return VariableSpecifications()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        return model