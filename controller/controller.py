import uuid

from controller.block_loader.block_loader import BlockLoader
from controller.component_type_loader.component_type_loader import ComponentTypeLoader
from helpers.xml.xml_helper import XmlHelper
from model.canvas.canvas_model import CanvasModel
from model.graph.graph_runners.python_graph_runner import GraphRunner
from model.session.session_model import SessionModel
from repository.canvas.canvas_loader import CanvasLoader
from repository.canvas.canvas_repository import CanvasRepository
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.component.component_loader import ComponentLoader
from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.component.component_type.component_type_specifications import ComponentTypeSpecifications
from repository.graph.edge_loader import EdgeLoader
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class Controller:

    session_model = None
    canvas_repository = None
    default_component_type_folder = "/home/michael/Projects/Mindblocks2.0/component_types"

    def __init__(self):
        identifier_repository = IdentifierRepository()
        self.canvas_repository = CanvasRepository(identifier_repository)
        self.graph_repository = GraphRepository(identifier_repository, self.canvas_repository)
        self.component_type_repository = ComponentTypeRepository(identifier_repository)
        self.component_type_loader = ComponentTypeLoader()
        self.component_repository = ComponentRepository(identifier_repository, self.canvas_repository, self.graph_repository, self.component_type_repository)

        xml_helper = XmlHelper()
        edge_loader = EdgeLoader(xml_helper, self.component_repository, self.graph_repository)
        component_loader = ComponentLoader(xml_helper, self.component_repository)
        canvas_loader = CanvasLoader(xml_helper, edge_loader, self.canvas_repository, component_loader)
        self.block_loader = BlockLoader(xml_helper, canvas_loader)

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
            if k == "id":
                v = uuid.UUID(v)
            specifications.add(k,v)

        return self.graph_repository.get(specifications)

    def get_component_types(self, specifications_dict={}):
        specifications = ComponentTypeSpecifications()
        for k,v in specifications_dict.items():
            specifications.add(k,v)

        return self.component_type_repository.get(specifications)

    def load_component_types(self, folder):
        component_types = self.component_type_loader.load_component_type_folder(folder)
        for component_type in component_types:
            self.component_type_repository.add(component_type)

        return component_types

    def load_default_component_types(self):
        return self.load_component_types(self.default_component_type_folder)

    def load_block_file(self, filename):
        return self.block_loader.load(filename)

    def run_graphs(self, specifications_dict={}):
        graph_runner = GraphRunner()
        graphs = self.get_graphs(specifications_dict)
        output = []
        for graph in graphs:
            output.append(graph_runner.run(graph, {}))
        return output
