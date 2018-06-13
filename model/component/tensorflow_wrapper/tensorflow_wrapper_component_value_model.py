from model.component.component_model import ComponentModel
from model.component.component_value_model import ComponentValueModel
from model.component.tensorflow_wrapper.tensorflow_placeholder_socket import TensorflowPlaceholderSocket


class TensorflowWrapperComponentValueModel(ComponentValueModel):

    inner_component = None
    graph = None
    variables = None

    def __init__(self, inner_component):
        self.inner_component = inner_component

    def compile_graph(self):
        self.variables = []
        for i in range(len(self.inner_component.in_sockets)):
            placeholder = TensorflowPlaceholderSocket(self.inner_component.in_sockets[i])
            self.variables.append(placeholder.value)

        self.graph = self.inner_component.component_type.execute(self.variables,
                                                                 self.inner_component.value,
                                                                 language="tensorflow")

    def get_variables(self):
        return self.variables

    def get_out_value_types(self):
        return self.inner_component.get_out_value_types()

    def initialize(self):
        self.inner_component.initialize()