from model.component.component_model import ComponentModel
from model.component.contraction.contraction_component_model import ContractionComponentModel
from model.component.tensorflow_wrapper.tensorflow_wrapper_component_model import TensorflowWrapperComponentModel


class GraphModel:

    vertices = None
    edges = None
    identifier = None
    name = None

    canvas_id = None
    canvas_name = None

    mirror = None

    def __init__(self):
        self.vertices = []
        self.edges = []

    def get_mirror(self):
        if self.mirror is None:
            self.mirror = self.copy()

        return self.mirror

    def copy(self):
        copy = GraphModel()
        copy.name = self.name

        unfilled_edges = {}
        for component in self.topological_walk():
            component_copy = component.copy()

            for out_socket in component.out_sockets:
                for edge in out_socket:
                    edge_copy = edge.copy()
                    copy.add_edge(edge_copy)

                    component_copy.out_sockets[edge.source_socket].append(edge_copy)
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

    def has_edges(self):
        return len(self.edges) > 0

    def get_components(self, component_type_name):
        l = []
        for component in self.vertices:
            if component.component_type.name == component_type_name:
                l.append(component)
        return l

    def __str__(self):
        return "Graph: "+str(self.name)+"|"+str(self.identifier) + " [" + " ".join([c.get_name() for c in self.vertices]) + "]"

    def describe(self):
        #TODO: Print edges
        return "Graph: "+str(self.name)+"|"+str(self.identifier) + " [" + " ".join([c.get_name() for c in self.vertices]) + "]"

    def initialize(self):
        for vertex in self.topological_walk():
            vertex.initialize()

    def topological_walk(self):
        S = [vertex for vertex in self.vertices if vertex.all_in_edges_satisfied()]

        while len(S) > 0:
            next_vertex = S.pop()

            # Propagate forward in the graph:
            for out_socket in next_vertex.matched_out_sockets():
                for out_edge in out_socket:
                    out_edge.mark_satisfied(True)
                    if out_edge.get_target().all_in_edges_satisfied():
                        S.append(out_edge.get_target())

            #if not (components_only and next_vertex.is_socket()):
            yield next_vertex

        # Prepare for next traversal:
        for vertex in self.vertices:
            for out_socket in vertex.matched_out_sockets():
                for out_edge in out_socket:
                    out_edge.mark_satisfied(False)

    def find_internal_tensorflow_edge(self):
        for edge in self.edges:
            if edge.source.language == "tensorflow" and edge.target.language == "tensorflow":
                return edge

    def compile_tensorflow(self):
        edge = self.find_internal_tensorflow_edge()
        while edge is not None:
            self.contract(edge)
            edge = self.find_internal_tensorflow_edge()

        replacers = []

        for component in self.vertices:
            if component.language == "tensorflow":
                tensorflow_wrapper = TensorflowWrapperComponentModel(component)
                replacers.append((component, tensorflow_wrapper))

        for component, replacer in replacers:
            self.replace(component, replacer)

    def get_compiled_copy(self):
        copy = self.copy()
        copy.compile_tensorflow()
        return copy

    def replace(self, component, other_component):
        self.vertices.remove(component)
        self.add_component(other_component)

        other_component.out_sockets = component.out_sockets
        other_component.in_sockets = component.in_sockets

        for edge in component.in_sockets:
            edge.target = other_component

        for socket in component.out_sockets:
            for edge in socket:
                edge.source = other_component

    def contract(self, edge):
        source = edge.source
        target = edge.target
        self.edges.remove(edge)
        self.vertices.remove(edge.source)
        self.vertices.remove(edge.target)

        meta_component = ContractionComponentModel(edge)

        if source.language == "tensorflow" and target.language == "tensorflow":
            meta_component.language = "tensorflow"

        self.add_component(meta_component)

        meta_component.in_sockets = []
        for prior_edge in source.in_sockets:
            meta_component.in_sockets.append(prior_edge)
            if prior_edge is not None:
                prior_edge.target = meta_component

        meta_component.out_sockets = []
        for subsequent_edge in target.out_sockets:
            meta_component.out_sockets.append(subsequent_edge)
            for sub_edge in subsequent_edge:
                sub_edge.source = meta_component

        for subsequent_edge in source.out_sockets:
            if len(subsequent_edge) == 0:
                meta_component.out_sockets.append(subsequent_edge)
            else:
                new_socket = []
                for sub_edge in subsequent_edge:
                    if sub_edge != edge and sub_edge.target == target:
                        meta_component.contract_additional_edge(sub_edge)
                        self.edges.remove(sub_edge)
                    elif sub_edge != edge:
                        new_socket.append(sub_edge)
                        sub_edge.source = meta_component
                        meta_component.add_source_output(sub_edge.source_socket)
                        sub_edge.source_socket = len(meta_component.out_sockets)

                if len(new_socket) > 0:
                    meta_component.out_sockets.append(new_socket)

        for prior_edge in target.in_sockets:
            if prior_edge != edge:
                if prior_edge is not None and prior_edge.source == source:
                    pass
                else:
                    meta_component.in_sockets.append(prior_edge)
                    if prior_edge is not None:
                        prior_edge.target = meta_component
                        prior_edge.target_socket = len(meta_component.in_sockets) - 1

        return meta_component
