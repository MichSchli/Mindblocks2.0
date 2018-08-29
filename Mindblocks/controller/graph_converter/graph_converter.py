from Mindblocks.controller.graph_converter.execution_graph_builder import ExecutionGraphBuilder
from Mindblocks.controller.graph_converter.tensorflow_section_contractor import TensorflowSectionContractor
from Mindblocks.controller.graph_converter.value_dictionary_builder import ValueDictionaryBuilder
from Mindblocks.model.execution_graph.execution_component_model import ExecutionComponentModel
from Mindblocks.model.execution_graph.execution_graph_model import ExecutionGraphModel
from Mindblocks.model.execution_graph.execution_head_component import ExecutionHeadComponent
from Mindblocks.model.execution_graph.execution_in_socket import ExecutionInSocket
from Mindblocks.model.execution_graph.execution_out_socket import ExecutionOutSocket
from Mindblocks.repository.graph.graph_specifications import GraphSpecifications


class GraphConverter:

    tensorflow_section_contractor = None
    variable_repository = None
    graph_repository = None

    def __init__(self, variable_repository, graph_repository, tensorflow_session_repository, execution_component_repository):
        self.tensorflow_section_contractor = TensorflowSectionContractor(tensorflow_session_repository)
        self.variable_repository = variable_repository
        self.graph_repository = graph_repository

        self.value_dictionary_builder = ValueDictionaryBuilder(variable_repository, graph_repository)
        self.execution_graph_builder = ExecutionGraphBuilder(graph_repository, execution_component_repository, variable_repository)

    def to_executable(self, runs, run_modes=None, tensorflow_session_model=None):
        if run_modes is None:
            run_modes = ["test" for _ in runs]

        value_dictionary = self.value_dictionary_builder.build_value_dictionary(runs, run_modes)

        execution_graphs = []

        for run, mode in zip(runs, run_modes):
            run_graph = self.execution_graph_builder.build_execution_graph(run, mode, value_dictionary)
            run_graph.initialize_type_models()
            execution_graphs.append(run_graph)

        self.tensorflow_section_contractor.contract_tensorflow_sections_in_graphs(execution_graphs,
                                                                                  run_modes,
                                                                                  tensorflow_session_model=tensorflow_session_model)

        return execution_graphs

    def should_populate(self, value):
        for populate_key, populate_spec_dict in value.get_populate_items():
            if populate_key == "graph":
                return True

        return False
