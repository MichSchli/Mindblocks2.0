from Mindblocks.controller.block_loader.block_loader import BlockLoader
from Mindblocks.controller.block_loader.canvas_loader import CanvasLoader
from Mindblocks.controller.block_loader.component_loader import ComponentLoader
from Mindblocks.controller.block_loader.configuration_loader import ConfigurationLoader
from Mindblocks.controller.block_loader.edge_loader import EdgeLoader
from Mindblocks.controller.block_loader.graph_loader import GraphLoader
from Mindblocks.controller.block_loader.variable_loader import VariableLoader
from Mindblocks.controller.component_type_loader.component_type_loader import ComponentTypeLoader
from Mindblocks.controller.graph_converter.graph_converter import GraphConverter
from Mindblocks.controller.ml_helper.ml_helper_factory import MlHelperFactory
from Mindblocks.helpers.files.FilepathHandler import FilepathHandler
from Mindblocks.helpers.xml.xml_helper import XmlHelper
from Mindblocks.repository.canvas_repository.canvas_repository import CanvasRepository
from Mindblocks.repository.component_type_repository.component_type_repository import ComponentTypeRepository
from Mindblocks.repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from Mindblocks.repository.graph.graph_repository import GraphRepository
from Mindblocks.repository.identifier.identifier_repository import IdentifierRepository
from Mindblocks.repository.variable_repository.variable_repository import VariableRepository
import tensorflow as tf


class SetupHolder:

    def __init__(self, load_default_types=True):
        tf.reset_default_graph()

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
        self.graph_loader = GraphLoader(self.xml_helper, self.component_loader, self.edge_loader,
                                        self.graph_repository)
        self.canvas_loader = CanvasLoader(self.xml_helper, self.component_loader, self.edge_loader, self.graph_loader,
                                          self.canvas_repository)

        self.variable_repository = VariableRepository(self.identifier_repository)
        self.variable_loader = VariableLoader(self.xml_helper, self.variable_repository)
        self.configuration_loader = ConfigurationLoader(self.xml_helper, self.variable_loader)

        self.block_loader = BlockLoader(self.xml_helper, self.canvas_loader, self.configuration_loader)

        self.graph_converter = GraphConverter(self.variable_repository, self.graph_repository)

        self.ml_helper_factory = MlHelperFactory(self.graph_converter, self.variable_repository)