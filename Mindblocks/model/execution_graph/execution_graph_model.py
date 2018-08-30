class ExecutionGraphModel:

    head_component = None
    components = None
    run_mode = None

    def __init__(self):
        self.components = []

    def add_head_component(self, head_component):
        self.head_component = head_component

    def execute(self, discard_value_models=True):
        self.clear_all_caches()
        return [v.get_value() if discard_value_models else v for v in self.head_component.pull(self.run_mode)]

    def clear_all_caches(self):
        self.head_component.clear_caches()

    def count_components(self):
        return len(self.components)

    def get_components(self):
        return self.components

    def add_execution_component(self, execution_component):
        self.components.append(execution_component)

    def initialize(self):
        self.clear_all_caches()
        self.head_component.initialize(self.run_mode, self.tensorflow_session_model)

    def init_batches(self):
        self.clear_all_caches()
        self.head_component.init_batches()

    def has_batches(self):
        return self.head_component.has_batches()

    def get_out_socket(self, component_name, socket_name):
        for component in self.components:
            if component.get_name() == component_name:
                return component.out_sockets[socket_name]

    def get_in_socket(self, component_name, socket_name):
        for component in self.components:
            if component.get_name() == component_name:
                return component.in_sockets[socket_name]

    def enforce_value(self, component, socket, value):
        in_socket = self.get_in_socket(component, socket)
        in_socket.replace_value(value)

    def enforce_type(self, component, socket, type):
        in_socket = self.get_in_socket(component, socket)
        in_socket.replace_type(type)

    def initialize_type_models(self, mode):
        return self.head_component.initialize_type_models(mode)

    def count_parameters(self):
        parameters = 0
        for component in self.components:
            parameters += component.count_parameters()

        for value in self.referenced_graph_values():
            parameters += value.count_parameters()

        return parameters

    def referenced_graph_values(self):
        values = []

        for component in self.components:
            for referenced_graph in component.get_referenced_graphs():
                values.extend(referenced_graph.get_all_values())

        values = list(set(values))
        return values

    def get_all_values(self):
        return [component.get_value() for component in self.components]