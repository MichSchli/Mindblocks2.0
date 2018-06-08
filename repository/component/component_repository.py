from model.component.component_model import ComponentModel
from repository.abstract_repository import AbstractRepository
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.component.component_type.component_type_specifications import ComponentTypeSpecifications
from repository.graph.graph_specifications import GraphSpecifications


class ComponentRepository(AbstractRepository):

    def __init__(self, identifier_repository, canvas_repository, graph_repository, component_type_repository):
        AbstractRepository.__init__(self, identifier_repository)
        self.canvas_repository = canvas_repository
        self.graph_repository = graph_repository
        self.component_type_repository = component_type_repository

    def create(self, specifications):
        component = ComponentModel()
        component.name = specifications.name
        component.identifier = self.identifier_repository.create()

        if specifications.canvas_id is not None or specifications.canvas_name is not None:
            canvas_specifications = CanvasSpecifications()
            canvas_specifications.name = specifications.canvas_name
            canvas_specifications.identifier = specifications.canvas_id
            canvas = self.canvas_repository.get(canvas_specifications)[0]
            canvas.add_component(component)

            component.canvas_id = canvas.identifier
            component.canvas_name = canvas.name

        graph_specifications = GraphSpecifications()
        if specifications.canvas_id is not None or specifications.canvas_name is not None:
            graph_specifications.canvas_name = specifications.canvas_name
            graph_specifications.canvas_id = specifications.canvas_id

        if specifications.component_type_name is not None or specifications.component_type_id is not None:
            component_type_specifications = ComponentTypeSpecifications()
            component_type_specifications.name = specifications.component_type_name
            component_type_specifications.identifier = specifications.component_type_id
            component.component_type = self.component_type_repository.get(component_type_specifications)[0]

            component.value = component.component_type.get_new_value()
            component.in_sockets = [None] * component.component_type.in_degree()
            component.out_sockets = [None] * component.component_type.out_degree()

        if specifications.graph_id is None:
            graph = self.graph_repository.create(graph_specifications)
            graph.add_component(component)
            component.graph_id = graph.identifier
        else:
            component.graph_id = specifications.graph_id

        self.__create__(component)

        return component