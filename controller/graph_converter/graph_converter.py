from controller.graph_converter.tensorflow_section_contractor import TensorflowSectionContractor
from model.execution_graph.execution_component_model import ExecutionComponentModel
from model.execution_graph.execution_graph_model import ExecutionGraphModel
from model.execution_graph.execution_head_component import ExecutionHeadComponent
from model.execution_graph.execution_in_socket import ExecutionInSocket
from model.execution_graph.execution_out_socket import ExecutionOutSocket


class GraphConverter:

    tensorflow_section_contractor = None

    def __init__(self):
        self.tensorflow_section_contractor = TensorflowSectionContractor()

    def to_executable(self, runs):
        value_dictionary = self.build_value_dictionary(runs)

        execution_graphs = []

        for run in runs:
            run_graph = self.build_execution_graph(run, value_dictionary)

            self.tensorflow_section_contractor.contract_tensorflow_sections(run_graph)

            execution_graphs.append(run_graph)

        return execution_graphs

    def build_execution_graph(self, run, value_dictionary):
        head_component, execution_components = self.get_run_components_and_edges(run, value_dictionary)
        run_graph = ExecutionGraphModel()
        run_graph.add_head_component(head_component)

        for execution_component in execution_components:
            run_graph.add_execution_component(execution_component)

        return run_graph

    def build_value_dictionary(self, runs):
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

    def get_run_components_and_edges(self, run, value_dictionary):
        run_output_socket_ids = [str(socket.component.identifier) + ":" + socket.name for socket in run]

        activated_output_sockets = run[:]
        processed_components = []

        unmatched_in_sockets = {}
        execution_out_sockets = {}

        execution_components = []

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
                execution_components.append(execution_component)

                socket_id = str(component.identifier) + ":" + name
                execution_out_socket.socket_id = socket_id
                execution_out_sockets[socket_id] = execution_out_socket

            for name, socket in component.in_sockets.items():
                execution_in_socket = ExecutionInSocket()
                execution_component.add_in_socket(name, execution_in_socket)
                execution_in_socket.execution_component = execution_component

                desired_source_id = str(socket.edge.source_socket.component.identifier) + ":" + socket.edge.source_socket.name
                if desired_source_id not in unmatched_in_sockets:
                    unmatched_in_sockets[desired_source_id] = []

                unmatched_in_sockets[desired_source_id].append(execution_in_socket)

                activated_output_sockets.append(socket.edge.source_socket)

        for execution_out_socket in list(execution_out_sockets.values()):

            if execution_out_socket.socket_id in unmatched_in_sockets:
                for in_socket in unmatched_in_sockets[execution_out_socket.socket_id]:
                    in_socket.set_source(execution_out_socket)
                    execution_out_socket.add_target(in_socket)

        head_component = ExecutionHeadComponent()

        for socket_id in run_output_socket_ids:
            socket = execution_out_sockets[socket_id]

            head_in_socket = ExecutionInSocket()
            head_in_socket.set_source(socket)
            socket.add_target(head_in_socket)
            head_in_socket.execution_component = head_component

            head_component.add_in_socket(head_in_socket)

        return head_component, execution_components

    def build_execution_component(self, component, execution_value):
        execution_component_model = ExecutionComponentModel()
        execution_component_model.execution_value = execution_value
        execution_component_model.execution_type = component.component_type
        execution_component_model.identifier = component.identifier
        execution_component_model.language = component.language
        return execution_component_model