from model.component.component_model import ComponentModel
from model.component.tensorflow_wrapper.tensorflow_placeholder_socket import TensorflowPlaceholderSocket


class TensorflowWrapperComponentValueModel(ComponentModel):

    inner_component = None
    graph = None

    def __init__(self, inner_component):
        self.inner_component = inner_component
        self.compile_graph()

    def compile_graph(self):
        for i in range(len(self.inner_component.in_sockets)):
            self.inner_component.in_sockets[i] = TensorflowPlaceholderSocket()
        self.graph = self.inner_component.run(language="tensorflow")