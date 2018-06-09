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

    def copy(self):
        copy = GraphModel()
        copy.name = self.name

        unfilled_edges = {}
        for component in self.topological_walk():
            component_copy = component.copy()

            for edge in component.out_sockets:
                edge_copy = edge.copy()
                copy.add_edge(edge_copy)

                component_copy.out_sockets[edge.source_socket] = edge_copy
                edge_copy.source = component_copy

                target_id = edge.target.identifier
                target_socket = edge.target_socket

                key = str(target_id) + str(target_socket)

                unfilled_edges[key] = edge_copy

            for i in range(len(component.in_sockets)):
                key = str(component.identifier) + str(i)
                if key in unfilled_edges:
                    edge_copy = unfilled_edges[key]
                    component_copy.in_sockets[i] = edge_copy
                    edge_copy.target = component_copy

            copy.add_component(component_copy)

        return copy


    def get_name(self):
        return self.name

    def add_component(self, component):
        self.vertices.append(component)

    def add_edge(self, edge):
        self.edges.append(edge)

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
                if out_edge.get_target().all_in_edges_satisfied():
                    S.append(out_edge.get_target())

            #if not (components_only and next_vertex.is_socket()):
            yield next_vertex

        # Prepare for next traversal:
        for vertex in self.vertices:
            for out_edge in vertex.matched_out_sockets():
                out_edge.mark_satisfied(False)