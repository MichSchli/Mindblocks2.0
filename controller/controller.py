from model.canvas.canvas_model import CanvasModel
from model.session.session_model import SessionModel
from repository.canvas.canvas_repository import CanvasRepository
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class Controller:

    session_model = None
    canvas_repository = None

    def __init__(self):
        identifier_repository = IdentifierRepository()
        self.canvas_repository = CanvasRepository(identifier_repository)
        self.graph_repository = GraphRepository(identifier_repository, self.canvas_repository)
        self.component_type_repository = ComponentTypeRepository(identifier_repository)
        self.component_repository = ComponentRepository(identifier_repository, self.canvas_repository, self.graph_repository, self.component_type_repository)

    def initialize_model(self):
        session_model = SessionModel()
        self.session_model = session_model
        return session_model

    def add_canvas(self, specification_dict={}):
        specifications = CanvasSpecifications()
        specifications.name = specification_dict["name"] if "name" in specification_dict else None
        canvas = self.canvas_repository.create(specifications)

        self.session_model.add_canvas(canvas)

    def add_component(self, specification_dict={}):
        specifications = ComponentSpecifications()
        for k,v in specification_dict.items():
            specifications.add(k,v)

        self.component_repository.create(specifications)

    def add_graph(self, specification_dict={}):
        specifications = GraphSpecifications()
        for k,v in specification_dict.items():
            specifications.add(k,v)

        self.graph_repository.create(specifications)

    def get_canvases(self, specifications_dict={}):
        specifications = CanvasSpecifications()
        for k,v in specifications_dict.items():
            specifications.add(k,v)

        return self.canvas_repository.get(specifications)

    def get_components(self, specifications_dict={}):
        specifications = ComponentSpecifications()
        for k,v in specifications_dict.items():
            specifications.add(k,v)

        return self.component_repository.get(specifications)

    def get_graphs(self, specifications_dict={}):
        specifications = GraphSpecifications()
        for k,v in specifications_dict.items():
            specifications.add(k,v)

        return self.graph_repository.get(specifications)