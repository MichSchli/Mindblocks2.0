class GraphReferenceCollector:

    graph_repository = None
    execution_graph_builder = None
    value_dictionary_builder = None

    def __init__(self, graph_repository, execution_graph_builder, value_dictionary_builder):
        self.graph_repository = graph_repository
        self.execution_graph_builder = execution_graph_builder
        self.value_dictionary_builder = value_dictionary_builder

    def collect(self, execution_graphs):
        components = []
        for g in execution_graphs:
            components.extend(g.get_components())

        while len(components) > 0:
            component = components.pop()
            mode = component.get_mode()
            referenced_graph_outputs = self.retrieve_referenced_graph_outputs(component, mode)

            if len(referenced_graph_outputs) == 0:
                continue

            ref_g = self.execution_graph_builder.build_execution_graph(referenced_graph_outputs, mode)
            self.value_dictionary_builder.initialize_values([ref_g])

            component.value_model.set_graph(ref_g)
            components.extend(ref_g.get_components())

    def retrieve_referenced_graph_outputs(self, component, mode):
        if not component.has_referenced_graphs(mode):
            return []

        graph_name, component_names, socket_names = component.get_referenced_sockets(mode)
        graph = self.graph_repository.get_by_name(graph_name)[0]

        sockets = []
        for component_name, socket_name in zip(component_names, socket_names):
            socket = graph.get_out_socket(component_name, socket_name)
            sockets.append(socket)

        return sockets
