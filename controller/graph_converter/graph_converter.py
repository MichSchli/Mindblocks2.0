from model.execution_graph.execution_component_model import ExecutionComponentModel
from model.execution_graph.execution_graph_model import ExecutionGraphModel
from model.execution_graph.execution_head_component import ExecutionHeadComponent
from model.execution_graph.execution_in_socket import ExecutionInSocket
from model.execution_graph.execution_out_socket import ExecutionOutSocket


class GraphConverter:

    def to_executable(self, creation_graph, runs):
        value_dictionary = self.build_value_dictionary(creation_graph, runs)

        execution_graphs = []

        for run in runs:
            run_graph = self.build_execution_graph(creation_graph, run, value_dictionary)

            #self.contract_tensorflow_sections(run_graph)

            execution_graphs.append(run_graph)

        return execution_graphs

    def build_execution_graph(self, creation_graph, run, value_dictionary):
        head_component = self.get_run_components_and_edges(creation_graph, run, value_dictionary)
        run_graph = ExecutionGraphModel()
        run_graph.add_head_component(head_component)
        return run_graph

    def build_value_dictionary(self, creation_graph, runs):
        value_dictionary = {}

        activated_output_sockets = []
        for run in runs:
            for socket in run:
                activated_output_sockets.append(socket)

        while len(activated_output_sockets) > 0:
            socket = activated_output_sockets.pop()
            component = socket.get_component()

            if component.identifier in value_dictionary:
                continue

            value_dictionary[component.identifier] = self.initialize_value(component)

            for in_socket in list(component.in_sockets.values()):
                edge = in_socket.edge
                source_socket = edge.source_socket
                activated_output_sockets.append(source_socket)

        return value_dictionary

    def initialize_value(self, component):
        return component.component_type.initialize_value(component.component_value)

    def get_run_components_and_edges(self, creation_graph, run, value_dictionary):
        run_output_socket_ids = [str(socket.component.identifier) + ":" + socket.name for socket in run]

        activated_output_sockets = run[:]
        processed_components = []

        unmatched_in_sockets = {}
        execution_out_sockets = {}

        while len(activated_output_sockets) > 0:
            socket = activated_output_sockets.pop()
            component = socket.get_component()

            if component.identifier in processed_components:
                continue

            processed_components.append(component.identifier)

            execution_value = value_dictionary[component.identifier]
            execution_component = self.build_execution_component(component, execution_value)

            for name, socket in component.out_sockets.items():
                execution_out_socket = ExecutionOutSocket()
                execution_component.add_out_socket(name, execution_out_socket)
                execution_out_socket.execution_component = execution_component

                socket_id = str(component.identifier) + ":" + name
                execution_out_socket.socket_id = socket_id
                execution_out_sockets[socket_id] = execution_out_socket

            for name, socket in component.in_sockets.items():
                execution_in_socket = ExecutionInSocket()
                execution_component.add_in_socket(name, execution_in_socket)

                desired_source_id = str(socket.edge.source_socket.component.identifier) + ":" + socket.edge.source_socket.name
                if desired_source_id not in unmatched_in_sockets:
                    unmatched_in_sockets[desired_source_id] = []

                unmatched_in_sockets[desired_source_id].append(execution_in_socket)

                activated_output_sockets.append(socket.edge.source_socket)

        for execution_out_socket in list(execution_out_sockets.values()):

            if execution_out_socket.socket_id in unmatched_in_sockets:
                for in_socket in unmatched_in_sockets[execution_out_socket.socket_id]:
                    in_socket.set_source(execution_out_socket)

        run_output_sockets = [execution_out_sockets[socket_id] for socket_id in run_output_socket_ids]
        head_component = ExecutionHeadComponent(run_output_sockets)

        return head_component

    def build_execution_component(self, component, execution_value):
        execution_component_model = ExecutionComponentModel()
        execution_component_model.execution_value = execution_value
        execution_component_model.execution_type = component.component_type
        return execution_component_model

    def contract_tensorflow_sections(self, execution_graph):
        tensorflow_section = self.find_tensorflow_section(execution_graph)
        while tensorflow_section is not None:
            self.replace_tensorflow_section(execution_graph, tensorflow_section)
            tensorflow_section = self.find_tensorflow_section(execution_graph)

    def find_tensorflow_section(self, execution_graph):
        pass

    def replace_tensorflow_section(self, execution_graph, tensorflow_section):
        pass