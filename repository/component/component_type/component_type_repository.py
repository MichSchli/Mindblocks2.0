from model.canvas.canvas_model import CanvasModel
from model.component.component_type.component_type_model import ComponentTypeModel
from repository.abstract_repository import AbstractRepository


class ComponentTypeRepository(AbstractRepository):

    def create(self, specifications):
        model = ComponentTypeModel()
        model.name = specifications.name
        model.identifier = self.identifier_repository.create()

        if specifications.available_languages is not None:
            model.available_languages = specifications.available_languages

        self.__create__(model)

        return model

    def add(self, model):
        if model.identifier is None:
            model.identifier = self.identifier_repository.create()

        self.__create__(model)
        return model