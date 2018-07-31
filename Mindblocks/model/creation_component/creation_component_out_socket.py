class CreationComponentOutSocket:

    component = None
    name = None
    edges = None
    mark = None

    def __init__(self, component, name):
        self.component = component
        self.name = name
        self.edges = []

    def get_component(self):
        return self.component

    def add_edge(self, edge):
        self.edges.append(edge)

    def place_mark(self, mark):
        self.mark = mark

    def marked(self):
        return self.mark is not None

    def get_mark(self):
        return self.mark