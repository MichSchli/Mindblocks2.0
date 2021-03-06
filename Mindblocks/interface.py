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
from Mindblocks.controller.parameter_searcher.parameter_searcher import ParameterSearcher
from Mindblocks.graphic_interface.graphic_interface import GraphicInterface
from Mindblocks.helpers.files.FilepathHandler import FilepathHandler
from Mindblocks.helpers.logging.logger_factory import LoggerFactory
from Mindblocks.helpers.logging.logger_manager import LoggerManager
from Mindblocks.helpers.xml.xml_helper import XmlHelper
from Mindblocks.repository.canvas_repository.canvas_repository import CanvasRepository
from Mindblocks.repository.component_type_repository.component_type_repository import ComponentTypeRepository
from Mindblocks.repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from Mindblocks.repository.execution_component_repository.execution_component_repository import \
    ExecutionComponentRepository
from Mindblocks.repository.graph_repository.graph_repository import GraphRepository
from Mindblocks.repository.graph_repository.graph_specifications import GraphSpecifications
from Mindblocks.repository.identifier_repository.identifier_repository import IdentifierRepository
from Mindblocks.repository.tensorflow_session_repository.tensorflow_session_repository import \
    TensorflowSessionRepository
from Mindblocks.repository.variable_repository.variable_repository import VariableRepository
import tensorflow as tf

class BasicInterface:

    ml_helper = None

    def __init__(self, load_default_types=True):
        tf.reset_default_graph()
        self.logger_factory = LoggerFactory()
        self.logger_manager = LoggerManager(self.logger_factory)
        self.logger_manager.add_default_console_logger()

        self.identifier_repository = IdentifierRepository()
        self.type_repository = ComponentTypeRepository(self.identifier_repository, self.logger_manager)
        self.canvas_repository = CanvasRepository(self.identifier_repository, self.logger_manager)
        self.graph_repository = GraphRepository(self.identifier_repository, self.logger_manager)
        self.creation_component_repository = CreationComponentRepository(self.identifier_repository,
                                                                         self.type_repository,
                                                                         self.canvas_repository,
                                                                         self.graph_repository,
                                                                         self.logger_manager)
        self.tensorflow_session_repository = TensorflowSessionRepository(self.identifier_repository, self.logger_manager)
        self.execution_component_repository = ExecutionComponentRepository(self.identifier_repository, self.logger_manager)

        self.filepath_handler = FilepathHandler()
        self.component_type_loader = ComponentTypeLoader(self.filepath_handler, self.type_repository)

        if load_default_types:
            self.component_type_loader.load_default_folder()

        self.xml_helper = XmlHelper()
        self.component_loader = ComponentLoader(self.xml_helper, self.creation_component_repository)
        self.edge_loader = EdgeLoader(self.xml_helper, self.graph_repository, self.creation_component_repository)
        self.graph_loader = GraphLoader(self.xml_helper, self.component_loader, self.edge_loader,
                                        self.graph_repository)
        self.canvas_loader = CanvasLoader(self.xml_helper, self.component_loader, self.edge_loader, self.graph_loader,
                                          self.canvas_repository)

        self.variable_repository = VariableRepository(self.identifier_repository, self.logger_manager)
        self.variable_loader = VariableLoader(self.xml_helper, self.variable_repository)
        self.configuration_loader = ConfigurationLoader(self.xml_helper, self.variable_loader)

        self.block_loader = BlockLoader(self.xml_helper, self.canvas_loader, self.configuration_loader)

        self.graph_converter = GraphConverter(self.variable_repository,
                                              self.graph_repository,
                                              self.tensorflow_session_repository,
                                              self.execution_component_repository,
                                              self.logger_manager)

        self.ml_helper_factory = MlHelperFactory(self.graph_converter,
                                                 self.variable_repository,
                                                 self.tensorflow_session_repository,
                                                 self.logger_manager)

        self.parameter_searcher = ParameterSearcher(self.variable_repository, self.ml_helper_factory, self.logger_manager)

    def set_variable(self, name, value, mode=None):
        self.variable_repository.set_variable_value(name, value, mode=mode)

    def initialize(self, profile=False, log_dir=None):
        graph_specs = GraphSpecifications()
        graph_specs.marked = True
        graph = self.graph_repository.get(graph_specs)[0]
        self.ml_helper = self.ml_helper_factory.build_ml_helper_from_graph(graph, profile=profile, log_dir=log_dir)
        self.ml_helper.initialize_model()

    def load_file(self, filename):
        self.block_loader.load(filename)

    def train(self, iterations=None):
        self.ml_helper.train(iterations=iterations)

    def evaluate(self):
        return self.ml_helper.evaluate()

    def validate(self):
        return self.ml_helper.validate()

    def predict(self):
        return self.ml_helper.predict()

    def save(self, filepath):
        self.ml_helper.save(filepath)

    def load(self, filepath):
        self.ml_helper.load(filepath)

    def add_file_logger(self, config, filepath):
        self.logger_manager.add_file_logger(config, filepath)

    def add_console_logger(self, config):
        self.logger_manager.add_console_logger(config)

    def count_search_options(self, greedy=True):
        return self.parameter_searcher.count_search_options(greedy=greedy)

    def search(self, greedy=True, minimize_valid_score=True):
        graph_specs = GraphSpecifications()
        graph_specs.marked = True
        graph = self.graph_repository.get(graph_specs)[0]
        if greedy:
            search_results = self.parameter_searcher.greedy_search(graph, minimize_valid_score)
        else:
            search_results = self.parameter_searcher.grid_search(graph, minimize_valid_score)

        return search_results

    def apply_search_configuration(self, search_configuration, minimize_valid_score=True):
        graph_specs = GraphSpecifications()
        graph_specs.marked = True
        graph = self.graph_repository.get(graph_specs)[0]
        self.variable_repository.apply_search_configuration(search_configuration)
        self.ml_helper = self.ml_helper_factory.build_ml_helper_from_graph(graph, minimize_valid_score=minimize_valid_score)
        self.ml_helper.initialize_model()

    def get_execution_component(self, name):
        spec = self.execution_component_repository.get_specifications()
        spec.name = name
        return self.execution_component_repository.get(spec)[0]

    def get_component_type_repository(self):
        return self.type_repository

    def make_gui(self):
        gui = GraphicInterface(self)
        gui.initialize_view()
        gui.mainloop()
