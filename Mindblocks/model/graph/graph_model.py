from Mindblocks.model.abstract.abstract_model import AbstractModel


class GraphModel(AbstractModel):

    identifier = None
    name = None

    components = None
    edges = None

    def __init__(self):
        self.components = []
        self.edges = []

    def add_component(self, component):
        self.components.append(component)

    def count_vertices(self):
        return len(self.components)

    def get_vertices(self):
        return self.components

    def add_edge(self, edge):
        self.edges.append(edge)

    def count_edges(self):
        return len(self.edges)

    def get_edges(self):
        return self.edges

    def get_out_socket(self, component_name, socket_name):
        for component in self.components:
            if component.name == component_name:
                out_socket = component.get_out_socket(socket_name)
                return out_socket

    def get_in_socket(self, component_name, socket_name):
        for component in self.components:
            if component.name == component_name:
                in_socket = component.get_in_socket(socket_name)
                return in_socket

    def get_marked_sockets(self):
        marked_sockets = {}
        for component in self.components:
            for k,v in component.get_marked_sockets().items():
                marked_sockets[k] = v
        return marked_sockets

    def is_marked(self):
        for component in self.components:
            if len(component.get_marked_sockets()) > 0:
                return True
        return False

    def __str__(self):
        return self.name if self.name is not None else str(self.identifier)