from model.component.component_model import ComponentModel
from model.graph.graph_model import GraphModel
from repository.abstract_repository import AbstractRepository
from repository.canvas.canvas_specifications import CanvasSpecifications


class GraphRepository(AbstractRepository):

    def __init__(self, identifier_repository, canvas_repository):
        AbstractRepository.__init__(self, identifier_repository)
        self.canvas_repository = canvas_repository

    def create(self, specifications):
        graph = GraphModel()
        graph.name = specifications.name
        graph.identifier = self.identifier_repository.create()

        if specifications.canvas_id is not None or specifications.canvas_name is not None:
            canvas_specifications = CanvasSpecifications()
            canvas_specifications.name = specifications.canvas_name
            canvas_specifications.identifier = specifications.canvas_id
            canvas = self.canvas_repository.get(canvas_specifications)[0]
            canvas.add_graph(graph)

            graph.canvas_id = canvas.identifier
            graph.canvas_name = canvas.name

        self.__create__(graph)

        return graph

    def join_graphs(self, graph_1, graph_2):
        for component in graph_2.vertices:
            component.graph_id = graph_1.identifier
            graph_1.add_component(component)

        self.elements.remove(graph_2)

        if graph_2.canvas_id is not None or graph_2.canvas_name is not None:
            canvas_specifications = CanvasSpecifications()
            canvas_specifications.name = graph_2.canvas_name
            canvas_specifications.identifier = graph_2.canvas_id
            canvas = self.canvas_repository.get(canvas_specifications)[0]

            canvas.graphs.remove(graph_2)
