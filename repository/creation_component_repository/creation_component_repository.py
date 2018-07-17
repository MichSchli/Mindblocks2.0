from model.creation_component.creation_component_model import CreationComponentModel
from repository.abstract.abstract_repository import AbstractRepository
from repository.canvas_repository.canvas_specifications import CanvasSpecifications
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications


class CreationComponentRepository(AbstractRepository):

    component_type_repository = None

    def __init__(self, identifier_repository, component_type_repository, canvas_repository):
        AbstractRepository.__init__(self, identifier_repository)
        self.component_type_repository = component_type_repository
        self.canvas_repository = canvas_repository

    def __initialize_model__(self):
        return CreationComponentModel()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        self.assign_component_type(model, specifications)
        self.assign_canvas(model, specifications)

        model.component_value = {}
        if model.component_type is not None:
            model.component_type.assign_default_value(model.component_value)

        return model

    def assign_component_type(self, model, specifications):
        if specifications.component_type_name is not None:
            type_specs = ComponentTypeSpecifications()
            type_specs.name = specifications.component_type_name
            component_type = self.component_type_repository.get(type_specs)[0]
            model.component_type = component_type

    def assign_canvas(self, model, specifications):
        if specifications.canvas_name is not None or specifications.canvas_id is not None:
            canvas_specs = CanvasSpecifications()
            canvas_specs.name = specifications.canvas_name
            canvas_specs.identifier = specifications.canvas_id
            canvas = self.canvas_repository.get(canvas_specs)[0]
            model.canvas = canvas
            canvas.add_component(model)