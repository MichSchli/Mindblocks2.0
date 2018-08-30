from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository


class ComponentTypeRepository(AbstractRepository):

    def __initialize_model__(self):
        return ComponentTypeModel()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        return model

    def create_from_class(self, component_type_class):
        model = component_type_class()
        self.__fill__(model)
        self.add(model)