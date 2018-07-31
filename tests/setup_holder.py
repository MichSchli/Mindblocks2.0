from controller.block_loader.block_loader import BlockLoader
from controller.block_loader.canvas_loader import CanvasLoader
from controller.block_loader.component_loader import ComponentLoader
from controller.block_loader.configuration_loader import ConfigurationLoader
from controller.block_loader.edge_loader import EdgeLoader
from controller.block_loader.variable_loader import VariableLoader
from controller.component_type_loader.component_type_loader import ComponentTypeLoader
from controller.graph_converter.graph_converter import GraphConverter
from controller.ml_helper.ml_helper_factory import MlHelperFactory
from helpers.files.FilepathHandler import FilepathHandler
from helpers.xml.xml_helper import XmlHelper
from repository.canvas_repository.canvas_repository import CanvasRepository
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.graph.graph_repository import GraphRepository
from repository.identifier.identifier_repository import IdentifierRepository
from repository.variable_repository.variable_repository import VariableRepository


class SetupHolder:

    def __init__(self, load_default_types=True):
        self.identifier_repository = IdentifierRepository()
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.canvas_repository = CanvasRepository(self.identifier_repository)
        self.graph_repository = GraphRepository(self.identifier_repository)
        self.component_repository = CreationComponentRepository(self.identifier_repository,
                                                                self.type_repository,
                                                                self.canvas_repository,
                                                                self.graph_repository)

        self.filepath_handler = FilepathHandler()
        self.component_type_loader = ComponentTypeLoader(self.filepath_handler, self.type_repository)

        if load_default_types:
            self.component_type_loader.load_default_folder()

        self.xml_helper = XmlHelper()
        self.component_loader = ComponentLoader(self.xml_helper, self.component_repository)
        self.edge_loader = EdgeLoader(self.xml_helper, self.graph_repository, self.component_repository)
        self.canvas_loader = CanvasLoader(self.xml_helper, self.component_loader, self.edge_loader,
                                          self.canvas_repository)

        self.variable_repository = VariableRepository(self.identifier_repository)
        self.variable_loader = VariableLoader(self.xml_helper, self.variable_repository)
        self.configuration_loader = ConfigurationLoader(self.xml_helper, self.variable_loader)

        self.block_loader = BlockLoader(self.xml_helper, self.canvas_loader, self.configuration_loader)

        self.graph_converter = GraphConverter(self.variable_repository)

        self.ml_helper_factory = MlHelperFactory(self.graph_converter, self.variable_repository)