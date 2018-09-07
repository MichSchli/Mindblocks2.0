from Mindblocks.model.graph.edge_model import EdgeModel
from Mindblocks.model.graph.graph_model import GraphModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository
from Mindblocks.repository.graph_repository.graph_specifications import GraphSpecifications


class GraphRepository(AbstractRepository):

    def __initialize_model__(self):
        return GraphModel()

    def get_specifications(self):
        return GraphSpecifications()

    def create(self, specifications):
        model = self.__create__()
        model.name = specifications.name

        return model

    def merge(self, graph_1, graph_2):
        self.delete(graph_2)

        for component in graph_2.get_vertices():
            graph_1.add_component(component)
            component.graph = graph_1

        for edge in graph_2.get_edges():
            graph_1.add_edge(edge)

    def add_edge(self, out_socket, in_socket, cast=None):
        g1 = out_socket.get_component().graph
        g2 = in_socket.get_component().graph

        if g1 != g2:
            self.merge(g1, g2)

        edge = EdgeModel(out_socket, in_socket)
        edge.identifier = self.identifier_repository.create()

        g1.add_edge(edge)
        out_socket.add_edge(edge)
        in_socket.set_edge(edge)
        edge.cast = cast

        return edge