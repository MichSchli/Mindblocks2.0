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
        return "Graph: "+str(self.name)+"|"+str(self.identifier) + " [" + " ".join([c.get_name() for c in self.vertices]) + "]"

    def describe(self):
        #TODO: Print edges
        return "Graph: "+str(self.name)+"|"+str(self.identifier) + " [" + " ".join([c.get_name() for c in self.vertices]) + "]"

    def topological_walk(self, components_only=False):
        S = [vertex for vertex in self.vertices if vertex.all_in_edges_satisfied()]

        while len(S) > 0:
            next_vertex = S.pop()

            # Propagate forward in the graph:
            for out_edge in next_vertex.matched_out_sockets():
                out_edge.mark_satisfied(True)
                if out_edge.get_destination().all_in_edges_satisfied():
                    S.append(out_edge.get_destination())

            #if not (components_only and next_vertex.is_socket()):
            yield next_vertex

        # Prepare for next traversal:
        for vertex in self.vertices:
            for out_edge in vertex.matched_out_sockets():
                out_edge.mark_satisfied(False)