class GraphModel:

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