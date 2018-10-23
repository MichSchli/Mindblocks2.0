from Mindblocks.model.abstract.abstract_execution_model import AbstractExecutionModel


class ExecutionGraphModel(AbstractExecutionModel):

    head_component = None
    components = None
    run_mode = None
    execution_objects = None

    def __init__(self):
        self.components = []
        self.execution_objects = []

    def get_execution_objects(self):
        return self.execution_objects

    def add_head_component(self, head_component):
        self.head_component = head_component

    def execute(self, discard_value_models=True):
        self.clear_all_caches()
        out_values = [v for v in self.head_component.pull(self.run_mode)]

        return [v.format_for_program_output() if discard_value_models else v for v in out_values]

    def clear_all_caches(self):
        self.head_component.clear_caches()

    def count_components(self):
        return len(self.components)

    def get_components(self):
        return self.components

    def get_all_components(self):
        components = self.get_components()[:]
        for component in components:
            components.extend(component.get_referenced_components())

        return list(set(components))

    def add_execution_component(self, execution_component):
        self.components.append(execution_component)

    def add_execution_object(self, execution_object):
        self.execution_objects.append(execution_object)

    def initialize(self):
        self.clear_all_caches()
        self.head_component.initialize(self.run_mode, self.tensorflow_session_model)

    def init_batches(self):
        self.clear_all_caches()
        self.head_component.init_batches()

    def has_batches(self, mode):
        return self.head_component.has_batches(mode)

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

    def initialize_type_models(self):
        return self.head_component.initialize_type_models(self.run_mode)

    def count_parameters(self):
        parameters = 0
        counted_values = []
        for component in self.components:
            v = component.get_value_model()
            if v not in counted_values:
                counted_values.append(v)
                parameters += component.count_parameters()

        for component in self.referenced_graph_components():
            v = component.get_value_model()
            if v not in counted_values:
                counted_values.append(v)
                parameters += component.count_parameters()

        return parameters

    def referenced_graph_components(self):
        values = []

        for component in self.components:
            for referenced_graph in component.get_referenced_graphs():
                values.extend(referenced_graph.get_components())

        return values

    def get_all_values(self):
        return [component.get_value_model() for component in self.components]