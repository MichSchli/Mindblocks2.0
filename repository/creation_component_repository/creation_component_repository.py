from model.creation_component.creation_component_model import CreationComponentModel
from repository.abstract.abstract_repository import AbstractRepository
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications


class CreationComponentRepository(AbstractRepository):

    component_type_repository = None

    def __init__(self, identifier_repository, component_type_repository):
        AbstractRepository.__init__(self, identifier_repository)
        self.component_type_repository = component_type_repository

    def __initialize_model__(self):
        return CreationComponentModel()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        self.assign_component_type(model, specifications)

        return model

    def assign_component_type(self, model, specifications):
        if specifications.component_type_name is not None:
            type_specs = ComponentTypeSpecifications()
            type_specs.name = specifications.component_type_name
            type = self.component_type_repository.get(type_specs)[0]
            model.component_type = type