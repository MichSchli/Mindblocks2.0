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
        value_dictionary = {}

        activated_output_sockets = []
        for run in runs:
            for socket in run:
                activated_output_sockets.append(socket)

        seen_sockets = activated_output_sockets[:]

        while len(activated_output_sockets) > 0:
            socket = activated_output_sockets.pop()
            component = socket.get_component()

            value_dictionary[component.identifier] = self.initialize_value(component)

            for in_socket in list(component.in_sockets.values()):
                edge = in_socket.edge
                source_socket = edge.source_socket

                if source_socket not in seen_sockets:
                    activated_output_sockets.append(source_socket)
                    seen_sockets.append(source_socket)

        return value_dictionary

    def initialize_value(self, component):
        return component.component_type.initialize_value(component.component_value)

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