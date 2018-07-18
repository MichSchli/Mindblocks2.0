from model.execution_graph.execution_graph_model import ExecutionGraphModel


class GraphConverter:

    def to_executable(self, creation_graph, runs):
        value_dictionary = self.build_value_dictionary(creation_graph, runs)

        execution_graphs = []

        for run in runs:
            run_graph = self.build_execution_graph(creation_graph, run, value_dictionary)

            self.contract_tensorflow_sections(run_graph)

            execution_graphs.append(run_graph)

        return execution_graphs

    def build_execution_graph(self, creation_graph, run, value_dictionary):
        run_components, run_edges = self.get_run_components_and_edges(creation_graph, run, value_dictionary)
        run_graph = ExecutionGraphModel()
        run_graph.add_components(run_components)
        run_graph.add_edges(run_edges)
        return run_graph

    def build_value_dictionary(self, creation_graph, runs):
        return {}

    def get_run_components_and_edges(self, creation_graph, run, value_dictionary):
        return [], []

    def contract_tensorflow_sections(self, execution_graph):
        tensorflow_section = self.find_tensorflow_section(execution_graph)
        while tensorflow_section is not None:
            self.replace_tensorflow_section(execution_graph, tensorflow_section)
            tensorflow_section = self.find_tensorflow_section(execution_graph)

    def find_tensorflow_section(self, execution_graph):
        pass

    def replace_tensorflow_section(self, execution_graph, tensorflow_section):
        pass