class CreationComponentInSocket:

    component = None
    name = None
    edge = None

    def __init__(self, component, name):
        self.component = component
        self.name = name

    def get_component(self):
        return self.component

    def set_edge(self, edge):
        self.edge = edge