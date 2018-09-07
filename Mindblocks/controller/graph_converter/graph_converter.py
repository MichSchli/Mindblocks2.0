from Mindblocks.controller.graph_converter.execution_graph_builder import ExecutionGraphBuilder
from Mindblocks.controller.graph_converter.graph_reference_collector import GraphReferenceCollector
from Mindblocks.controller.graph_converter.tensorflow_section_contractor import TensorflowSectionContractor
from Mindblocks.controller.graph_converter.value_dictionary_builder import ValueDictionaryBuilder


class GraphConverter:

    tensorflow_section_contractor = None
    variable_repository = None
    graph_repository = None
    logger_manager = None
    graph_reference_collector = None

    def __init__(self, variable_repository, graph_repository, tensorflow_session_repository, execution_component_repository, logger_manager):
        self.tensorflow_section_contractor = TensorflowSectionContractor(tensorflow_session_repository)
        self.variable_repository = variable_repository
        self.graph_repository = graph_repository
        self.logger_manager = logger_manager

        self.execution_graph_builder = ExecutionGraphBuilder(graph_repository, execution_component_repository, logger_manager)
        self.value_dictionary_builder = ValueDictionaryBuilder(variable_repository, graph_repository, logger_manager)
        self.graph_reference_collector = GraphReferenceCollector(graph_repository, self.execution_graph_builder, self.value_dictionary_builder)

    def to_executable(self, runs, run_modes=None, tensorflow_session_model=None):
        if run_modes is None:
            run_modes = ["test" for _ in runs]

        execution_graphs = self.build_execution_graphs(run_modes, runs)

        self.initialize_variables(execution_graphs)

        self.handle_referenced_graphs(execution_graphs)

        self.initialize_type_models(execution_graphs)

        self.build_tensorflow_sections(execution_graphs, run_modes, tensorflow_session_model)

        return execution_graphs

    def build_tensorflow_sections(self, execution_graphs, run_modes, tensorflow_session_model):
        self.logger_manager.log("Building tensorflow sections...", "graph_construction", "status")
        self.tensorflow_section_contractor.contract_tensorflow_sections_in_graphs(execution_graphs,
                                                                                  run_modes,
                                                                                  tensorflow_session_model=tensorflow_session_model)

    def initialize_type_models(self, execution_graphs):
        self.logger_manager.log("Initializing type models and performing type checking...", "graph_construction",
                                "status")
        for execution_graph in execution_graphs:
            execution_graph.initialize_type_models()

    def handle_referenced_graphs(self, execution_graphs):
        self.logger_manager.log("Managing referenced graphs...", "graph_construction", "status")
        self.graph_reference_collector.collect(execution_graphs)

    def initialize_variables(self, execution_graphs):
        self.logger_manager.log("Initializing values...", "graph_construction", "status")
        self.value_dictionary_builder.initialize_values(execution_graphs)

    def build_execution_graphs(self, run_modes, runs):
        self.logger_manager.log("Contructing " + str(len(runs)) + " execution graphs...", "graph_construction",
                                "status")
        execution_graphs = []
        for run, mode in zip(runs, run_modes):
            execution_graph = self.execution_graph_builder.build_execution_graph(run, mode)
            execution_graphs.append(execution_graph)
        return execution_graphs
