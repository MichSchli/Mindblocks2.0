class GraphModel:

    vertices = None
    edges = None
    identifier = None
    name = None

    canvas_id = None
    canvas_name = None

    def __init__(self):
        self.vertices = []
        self.edges = []

    def get_name(self):
        return self.name

    def add_component(self, component):
        self.vertices.append(component)

    def __str__(self):
        return "Component: "+str(self.name)+"|"+str(self.identifier) + " [" + " ".join([c.get_name() for c in self.vertices]) + "]"