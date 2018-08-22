from Mindblocks.error_handling.loading.component_type_not_found_exception import ComponentTypeNotFoundException
from Mindblocks.model.creation_component.creation_component_in_socket import CreationComponentInSocket
from Mindblocks.model.creation_component.creation_component_model import CreationComponentModel
from Mindblocks.model.creation_component.creation_component_out_socket import CreationComponentOutSocket
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository
from Mindblocks.repository.canvas_repository.canvas_specifications import CanvasSpecifications
from Mindblocks.repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from Mindblocks.repository.creation_component_repository.creation_component_specifications import \
    CreationComponentSpecifications
from Mindblocks.repository.graph.graph_specifications import GraphSpecifications


class CreationComponentRepository(AbstractRepository):

    component_type_repository = None
    graph_repository = None
    canvas_repository = None

    def __init__(self, identifier_repository, component_type_repository, canvas_repository, graph_repository):
        AbstractRepository.__init__(self, identifier_repository)
        self.component_type_repository = component_type_repository
        self.canvas_repository = canvas_repository
        self.graph_repository = graph_repository

    def __initialize_model__(self):
        return CreationComponentModel()

    def get_specifications(self):
        return CreationComponentSpecifications()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        try:
            self.assign_component_type(model, specifications)
        except ComponentTypeNotFoundException:
            raise

        self.assign_canvas(model, specifications)
        self.assign_graph(model, specifications)

        model.component_value = {}
        if model.component_type is not None:
            model.component_type.assign_default_value(model.component_value)

        if specifications.language is not None:
            model.language = specifications.language
        elif model.component_type is not None and model.component_type.languages is not None:
            model.language = model.component_type.languages[0]

        return model

    def assign_component_type(self, model, specifications):
        if specifications.component_type_name is not None:
            type_specs = ComponentTypeSpecifications()
            type_specs.name = specifications.component_type_name
            component_types = self.component_type_repository.get(type_specs)

            if len(component_types) == 0:
                raise ComponentTypeNotFoundException("Attempted to create component with undeclared type " + specifications.component_type_name + ".")

            component_type = component_types[0]
            model.component_type = component_type

            for out_socket_name in model.component_type.get_out_sockets():
                out_socket = CreationComponentOutSocket(model, out_socket_name)
                model.add_out_socket(out_socket)

            for in_socket_name in model.component_type.get_in_sockets():
                in_socket = CreationComponentInSocket(model, in_socket_name)
                model.add_in_socket(in_socket)

    def assign_canvas(self, model, specifications):
        if specifications.canvas_name is not None or specifications.canvas_id is not None:
            canvas_specs = CanvasSpecifications()
            canvas_specs.name = specifications.canvas_name
            canvas_specs.identifier = specifications.canvas_id
            canvas = self.canvas_repository.get(canvas_specs)[0]
            model.canvas = canvas
            canvas.add_component(model)

    def assign_graph(self, model, specifications):
        graph_spec = GraphSpecifications()

        if specifications.graph_id is not None:
            graph_spec.identifier = specifications.graph_id
            graph = self.graph_repository.get(graph_spec)[0]
        else:
            graph = self.graph_repository.create(graph_spec)

        graph.add_component(model)
        model.graph = graph