class CreationComponentOutSocket:

    component = None
    name = None
    edges = None

    def __init__(self, component, name):
        self.component = component
        self.name = name
        self.edges = []

    def get_component(self):
        return self.component

    def add_edge(self, edge):
        self.edges.append(edge)