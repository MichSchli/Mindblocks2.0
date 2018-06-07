class CanvasModel:

    name = None
    identifier = None

    components = None
    graphs = None

    def __init__(self):
        self.components = []
        self.graphs = []

    def get_name(self):
        return self.name

    def __str__(self):
        return "Canvas: "+self.name+"|"+str(self.identifier)

    def get_components(self):
        return self.components

    def add_component(self, component):
        self.components.append(component)

    def get_graphs(self):
        return self.graphs

    def add_graph(self, graph):
        self.graphs.append(graph)