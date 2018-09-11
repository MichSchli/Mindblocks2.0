from Mindblocks.model.execution_graph.execution_edge import ExecutionEdge
from Mindblocks.model.execution_graph.execution_graph_model import ExecutionGraphModel
from Mindblocks.model.execution_graph.execution_head_component import ExecutionHeadComponent
from Mindblocks.model.execution_graph.execution_in_socket import ExecutionInSocket
from Mindblocks.model.execution_graph.execution_out_socket import ExecutionOutSocket


class ExecutionGraphBuilder:

    graph_repository = None
    execution_component_repository = None
    variable_repository = None

    def __init__(self, graph_repository, execution_component_repository, logger_manager):
        self.graph_repository = graph_repository
        self.execution_component_repository = execution_component_repository
        self.logger_manager = logger_manager

    def build_execution_graph(self, run, mode):
        self.log_graph_construction_start(mode, run)

        run_graph = ExecutionGraphModel()
        head_component, execution_components = self.get_run_components_and_edges(run, run_graph, mode)
        run_graph.run_mode = mode
        run_graph.add_head_component(head_component)

        for execution_component in execution_components:
            run_graph.add_execution_component(execution_component)

        return run_graph

    def log_graph_construction_start(self, mode, run):
        component_names = reversed([c.get_description() for c in run])
        self.logger_manager.log(
            "Contructing execution graph with end sockets [" + ", ".join(component_names) + "] and mode " + mode + ".",
            "graph_construction", "status")

    def get_run_components_and_edges(self, run, run_graph, run_mode):
        run_output_socket_ids = [str(socket.component.identifier) + ":" + socket.name for socket in run]

        activated_output_sockets = run[:]
        processed_components = []

        unmatched_execution_edges = {}
        execution_out_sockets = {}

        execution_components = []

        while len(activated_output_sockets) > 0:
            socket = activated_output_sockets.pop()
            component = socket.get_component()

            if component.identifier in processed_components:
                continue

            processed_components.append(component.identifier)

            self.logger_manager.log("Adding component " + component.get_name(), "graph_construction", "component")

            execution_component = self.build_execution_component(component, run_mode)
            execution_components.append(execution_component)
            run_graph.add_execution_object(execution_component)

            for name, socket in component.out_sockets.items():
                execution_out_socket = self.build_execution_out_socket(execution_component, socket, run_mode)
                run_graph.add_execution_object(execution_out_socket)

                socket_id = str(component.identifier) + ":" + name
                execution_out_sockets[socket_id] = execution_out_socket

            for name, socket in component.in_sockets.items():
                if not self.should_use(name, component, run_mode):
                    continue

                execution_in_socket = self.build_execution_in_socket(execution_component, socket, run_mode)
                run_graph.add_execution_object(execution_in_socket)

                if socket.edge is not None:
                    execution_edge = self.build_execution_edge(execution_in_socket, socket.edge, run_mode)
                    run_graph.add_execution_object(execution_edge)

                    desired_source_id = str(socket.edge.source_socket.component.identifier) + ":" + socket.edge.source_socket.name
                    if desired_source_id not in unmatched_execution_edges:
                        unmatched_execution_edges[desired_source_id] = []
                    unmatched_execution_edges[desired_source_id].append(execution_edge)

                    activated_output_sockets.append(socket.edge.source_socket)

        self.match_out_sockets_to_edges(execution_out_sockets, unmatched_execution_edges)

        head_component = self.build_execution_head_components(execution_out_sockets, run_output_socket_ids, run_mode)

        return head_component, execution_components

    def match_out_sockets_to_edges(self, execution_out_sockets, unmatched_in_sockets):
        for socket_id, execution_out_socket in execution_out_sockets.items():
            if socket_id in unmatched_in_sockets:
                for execution_edge in unmatched_in_sockets[socket_id]:
                    execution_edge.set_source(execution_out_socket)
                    execution_out_socket.add_edge(execution_edge)

    def should_use(self, name, creation_component, mode):
        execution_type = creation_component.component_type
        return execution_type.is_used(name, mode)

    def build_execution_head_components(self, execution_out_sockets, run_output_socket_ids, mode):
        head_component = ExecutionHeadComponent()
        for socket_id in run_output_socket_ids:
            socket = execution_out_sockets[socket_id]

            head_in_socket = ExecutionInSocket()
            output_edge = self.build_execution_edge(head_in_socket, None, mode)
            head_in_socket.execution_component = head_component
            head_component.add_in_socket(head_in_socket)

            output_edge.set_source(socket)
            socket.add_edge(output_edge)

        return head_component

    def build_execution_component(self, component, mode):
        execution_component_model = self.execution_component_repository.create_from_creation_component(component)
        execution_component_model.set_origin(component)
        execution_component_model.set_mode(mode)
        return execution_component_model

    def build_execution_in_socket(self, execution_component, socket, mode):
        execution_in_socket = ExecutionInSocket()
        execution_in_socket.set_origin(socket)
        execution_component.add_in_socket(socket.get_name(), execution_in_socket)
        execution_in_socket.execution_component = execution_component
        execution_in_socket.set_mode(mode)
        return execution_in_socket

    def build_execution_out_socket(self, execution_component, socket, mode):
        execution_out_socket = ExecutionOutSocket()
        execution_out_socket.set_origin(socket)
        execution_component.add_out_socket(socket.get_name(), execution_out_socket)
        execution_out_socket.execution_component = execution_component
        execution_out_socket.set_mode(mode)
        return execution_out_socket

    def build_execution_edge(self, execution_in_socket, creation_edge, run_mode):
        execution_edge = ExecutionEdge()
        execution_edge.set_origin(creation_edge)
        execution_edge.set_mode(run_mode)
        execution_in_socket.add_edge(execution_edge)
        execution_edge.set_target(execution_in_socket)
        return execution_edge