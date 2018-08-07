from Mindblocks.model.execution_graph.execution_component_model import ExecutionComponentModel
from Mindblocks.model.execution_graph.execution_graph_model import ExecutionGraphModel
from Mindblocks.model.execution_graph.execution_head_component import ExecutionHeadComponent
from Mindblocks.model.execution_graph.execution_in_socket import ExecutionInSocket
from Mindblocks.model.execution_graph.execution_out_socket import ExecutionOutSocket
from Mindblocks.repository.graph.graph_specifications import GraphSpecifications


class ExecutionGraphBuilder:

    graph_repository = None

    def __init__(self, graph_repository):
        self.graph_repository = graph_repository

    def build_execution_graph(self, run, mode, value_dictionary):
        head_component, execution_components = self.get_run_components_and_edges(run, mode, value_dictionary)
        run_graph = ExecutionGraphModel()
        run_graph.run_mode = mode
        run_graph.add_head_component(head_component)

        for execution_component in execution_components:
            run_graph.add_execution_component(execution_component)

        return run_graph

    def should_populate(self, value):
        for populate_key, populate_spec_dict in value.get_populate_items():
            if populate_key == "graph":
                return True

        return False

    def do_populate(self, value, run_mode, value_dictionary):
        for populate_key, populate_spec_dict in value.get_populate_items():
            if populate_key == "graph":
                spec = GraphSpecifications()
                spec.add_all(populate_spec_dict)
                graph = self.graph_repository.get(spec)[0]

                run = []
                for component_name, socket_name in value.get_required_graph_outputs():
                    run.append(graph.get_out_socket(component_name, socket_name))

                execution_graph = self.build_execution_graph(run, run_mode, value_dictionary)

                value.graph = execution_graph

        return value

    def get_run_components_and_edges(self, run, run_mode, value_dictionary):
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

            component_vals = value_dictionary[component.identifier]

            if run_mode in component_vals:
                execution_value = component_vals[run_mode]
            else:
                execution_value = component_vals["default"]

            if self.should_populate(execution_value):
                execution_value = self.do_populate(execution_value, run_mode, value_dictionary)

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

                if socket.edge is not None:

                    execution_in_socket.cast = socket.edge.cast

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
        execution_component_model.name = component.name
        return execution_component_model