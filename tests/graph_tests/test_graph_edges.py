import unittest

from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestGraphEdges(unittest.TestCase):

    def test_add_edge_has_correct_source_and_target(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        graph = repository.create(specifications)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        component_repository = ComponentRepository(identifier_repository, None, repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.graph_id = graph.identifier
        component_1 = component_repository.create(specifications)
        component_2 = component_repository.create(specifications)

        component_1.out_sockets = [None]
        component_2.in_sockets = [None]

        edge = repository.create_edge(component_1, 0, component_2, 0)

        self.assertEqual(edge.source, component_1)
        self.assertEqual(edge.target, component_2)

    def test_add_within_graph_creates_edge(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        graph = repository.create(specifications)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        component_repository = ComponentRepository(identifier_repository, None, repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.graph_id = graph.identifier
        component_1 = component_repository.create(specifications)
        component_2 = component_repository.create(specifications)

        component_1.out_sockets = [None]
        component_2.in_sockets = [None]

        repository.create_edge(component_1, 0, component_2, 0)

        self.assertIsNotNone(graph.edges)
        self.assertEqual(len(graph.edges), 1)

    def test_add_within_graph_correct_sockets(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        graph = repository.create(specifications)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        component_repository = ComponentRepository(identifier_repository, None, repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.graph_id = graph.identifier
        component_1 = component_repository.create(specifications)
        component_2 = component_repository.create(specifications)

        component_1.out_sockets = [None]
        component_2.in_sockets = [None]

        repository.create_edge(component_1, 0, component_2, 0)

        self.assertIsNotNone(component_1.out_sockets[0])
        self.assertIsNotNone(component_2.in_sockets[0])

        edge = graph.edges[0]
        self.assertEqual(component_1.out_sockets[0], edge)
        self.assertEqual(component_2.in_sockets[0], edge)

    def test_add_between_graphs_joins_graphs(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        component_repository = ComponentRepository(identifier_repository, None, repository, component_type_repository)
        specifications = ComponentSpecifications()
        component_1 = component_repository.create(specifications)
        component_2 = component_repository.create(specifications)

        component_1.out_sockets = [None]
        component_2.in_sockets = [None]

        self.assertEqual(len(repository.elements), 2)

        repository.create_edge(component_1, 0, component_2, 0)

        self.assertEqual(len(repository.elements), 1)
        self.assertEqual(component_1.graph_id, component_2.graph_id)

        specifications = GraphSpecifications()
        specifications.identifier = component_1.graph_id
        graphs = repository.get(specifications)

        self.assertEqual(len(graphs), 1)

    def test_add_between_graphs_creates_edges(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        component_repository = ComponentRepository(identifier_repository, None, repository, component_type_repository)
        specifications = ComponentSpecifications()
        component_1 = component_repository.create(specifications)
        component_2 = component_repository.create(specifications)

        component_1.out_sockets = [None]
        component_2.in_sockets = [None]

        repository.create_edge(component_1, 0, component_2, 0)

        specifications = GraphSpecifications()
        specifications.identifier = component_1.graph_id
        graph = repository.get(specifications)[0]

        self.assertIsNotNone(graph.edges)
        self.assertEqual(len(graph.edges), 1)

    def test_add_between_graphs_correct_sockets(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        component_repository = ComponentRepository(identifier_repository, None, repository, component_type_repository)
        specifications = ComponentSpecifications()
        component_1 = component_repository.create(specifications)
        component_2 = component_repository.create(specifications)

        component_1.out_sockets = [None]
        component_2.in_sockets = [None]

        repository.create_edge(component_1, 0, component_2, 0)

        self.assertIsNotNone(component_1.out_sockets[0])
        self.assertIsNotNone(component_2.in_sockets[0])

        specifications = GraphSpecifications()
        specifications.identifier = component_1.graph_id
        graph = repository.get(specifications)[0]
        edge = graph.edges[0]

        self.assertEqual(component_1.out_sockets[0], edge)
        self.assertEqual(component_2.in_sockets[0], edge)