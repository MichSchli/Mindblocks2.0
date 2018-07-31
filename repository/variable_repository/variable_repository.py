from model.variable.variable_model import VariableModel
from repository.abstract.abstract_repository import AbstractRepository


class VariableRepository(AbstractRepository):

    def __initialize_model__(self):
        return VariableModel()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        return model