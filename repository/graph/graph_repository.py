from model.graph.edge_model import EdgeModel
from model.graph.graph_model import GraphModel
from repository.abstract.abstract_repository import AbstractRepository


class GraphRepository(AbstractRepository):

    def __initialize_model__(self):
        return GraphModel()

    def create(self, specifications):
        model = self.__create__()

        return model

    def merge(self, graph_1, graph_2):
        self.delete(graph_2)

        for component in graph_2.get_vertices():
            graph_1.add_component(component)
            component.graph = graph_1

        for edge in graph_2.get_edges():
            graph_1.add_edge(edge)

    def add_edge(self, out_socket, in_socket):
        g1 = out_socket.get_component().graph
        g2 = in_socket.get_component().graph

        if g1 != g2:
            self.merge(g1, g2)

        edge = EdgeModel(out_socket, in_socket)

        g1.add_edge(edge)
        out_socket.add_edge(edge)
        in_socket.set_edge(edge)