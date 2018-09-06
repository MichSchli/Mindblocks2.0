from Mindblocks.controller.graph_converter.execution_graph_builder import ExecutionGraphBuilder
from Mindblocks.controller.graph_converter.tensorflow_section_contractor import TensorflowSectionContractor
from Mindblocks.controller.graph_converter.value_dictionary_builder import ValueDictionaryBuilder
from Mindblocks.model.execution_graph.execution_component_model import ExecutionComponentModel
from Mindblocks.model.execution_graph.execution_graph_model import ExecutionGraphModel
from Mindblocks.model.execution_graph.execution_head_component import ExecutionHeadComponent
from Mindblocks.model.execution_graph.execution_in_socket import ExecutionInSocket
from Mindblocks.model.execution_graph.execution_out_socket import ExecutionOutSocket
from Mindblocks.repository.graph_repository.graph_specifications import GraphSpecifications


class GraphConverter:

    tensorflow_section_contractor = None
    variable_repository = None
    graph_repository = None
    logger_manager = None

    def __init__(self, variable_repository, graph_repository, tensorflow_session_repository, execution_component_repository, logger_manager):
        self.tensorflow_section_contractor = TensorflowSectionContractor(tensorflow_session_repository)
        self.variable_repository = variable_repository
        self.graph_repository = graph_repository
        self.logger_manager = logger_manager

        self.execution_graph_builder = ExecutionGraphBuilder(graph_repository, execution_component_repository, logger_manager)
        self.value_dictionary_builder = ValueDictionaryBuilder(variable_repository, graph_repository, logger_manager)

    def to_executable(self, runs, run_modes=None, tensorflow_session_model=None):
        if run_modes is None:
            run_modes = ["test" for _ in runs]

        self.logger_manager.log("Contructing " + str(len(runs)) + " execution graphs...", "graph_construction", "status")

        execution_graphs = []

        for run, mode in zip(runs, run_modes):
            execution_graph = self.execution_graph_builder.build_execution_graph(run, mode)
            execution_graphs.append(execution_graph)

        self.logger_manager.log("Initializing values...", "graph_construction", "status")

        self.value_dictionary_builder.initialize_values(execution_graphs)

        self.logger_manager.log("Managing referenced graphs...", "graph_construction", "status")

        self.logger_manager.log("Initializing type models and performing type checking...", "graph_construction", "status")

        for execution_graph in execution_graphs:
            execution_graph.initialize_type_models()

        self.logger_manager.log("Building tensorflow sections...", "graph_construction", "status")

        self.tensorflow_section_contractor.contract_tensorflow_sections_in_graphs(execution_graphs,
                                                                                  run_modes,
                                                                                  tensorflow_session_model=tensorflow_session_model)

        return execution_graphs

    def should_populate(self, value):
        for populate_key, populate_spec_dict in value.get_populate_items():
            if populate_key == "graph":
                return True

        return False
