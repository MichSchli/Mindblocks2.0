import unittest

from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestGraphContractions(unittest.TestCase):

    """

    """

    def test_contract_edge_meta_component_built_correctly(self):
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

        component_1.in_sockets = [None]
        component_1.out_sockets = [[]]

        component_2.in_sockets = [None]
        component_2.out_sockets = [[]]

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)

        meta_component = graph.contract(edge_1)
        self.assertEqual(meta_component.in_sockets, [None])
        self.assertEqual(meta_component.out_sockets, [[]])

    def test_contract_edge_meta_component_built_correctly_multiple_sockets(self):
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

        component_1.in_sockets = [None, None]
        component_1.out_sockets = [[], [], [], []]

        component_2.in_sockets = [None, None]
        component_2.out_sockets = [[], []]

        edge_1 = repository.create_edge(component_1, 2, component_2, 1)

        meta_component = graph.contract(edge_1)
        self.assertEqual(meta_component.in_sockets, [None, None, None])
        self.assertEqual(meta_component.out_sockets, [[], [], [], [], []])

    def test_contract_edge_meta_component_built_correctly_multiple_edges(self):
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

        component_1.in_sockets = [None, None]
        component_1.out_sockets = [[], [], [], []]

        component_2.in_sockets = [None, None]
        component_2.out_sockets = [[], []]

        edge_1 = repository.create_edge(component_1, 2, component_2, 1)
        edge_2 = repository.create_edge(component_1, 1, component_2, 0)

        meta_component = graph.contract(edge_1)
        self.assertEqual(meta_component.in_sockets, [None, None])
        self.assertEqual(meta_component.out_sockets, [[], [], [], []])

    def test_contract_edge_meta_value_built_correctly(self):
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

        component_1.in_sockets = [None]
        component_1.out_sockets = [[]]

        component_2.in_sockets = [None]
        component_2.out_sockets = [[]]

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)

        meta_component = graph.contract(edge_1)

        self.assertEqual(meta_component.value.source_component, component_1)
        self.assertEqual(meta_component.value.target_component, component_2)
        self.assertEqual(meta_component.value.source_sockets, [0])
        self.assertEqual(meta_component.value.target_sockets, [0])

    def test_contract_edge_meta_value_built_correctly_multiple_sockets(self):
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

        component_1.in_sockets = [None, None]
        component_1.out_sockets = [[], [], [], []]

        component_2.in_sockets = [None, None]
        component_2.out_sockets = [[], []]

        edge_1 = repository.create_edge(component_1, 2, component_2, 1)

        meta_component = graph.contract(edge_1)
        self.assertEqual(meta_component.value.source_component, component_1)
        self.assertEqual(meta_component.value.target_component, component_2)
        self.assertEqual(meta_component.value.source_sockets, [2])
        self.assertEqual(meta_component.value.target_sockets, [1])


    def test_contract_edge_meta_value_built_correctly_multiple_edges(self):
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

        component_1.in_sockets = [None, None]
        component_1.out_sockets = [[], [], [], []]

        component_2.in_sockets = [None, None]
        component_2.out_sockets = [[], []]

        edge_1 = repository.create_edge(component_1, 2, component_2, 1)
        edge_2 = repository.create_edge(component_1, 1, component_2, 0)

        meta_component = graph.contract(edge_1)

        self.assertEqual(meta_component.value.source_component, component_1)
        self.assertEqual(meta_component.value.target_component, component_2)
        self.assertEqual(meta_component.value.source_sockets, [2, 1])
        self.assertEqual(meta_component.value.target_sockets, [1, 0])


    """
    s -> (contract) t -> u
    """
    def test_contract_edge_with_subsequent_graph(self):
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
        component_3 = component_repository.create(specifications)

        component_1.in_sockets = []
        component_1.out_sockets = [[]]

        component_2.in_sockets = [None]
        component_2.out_sockets = [[]]

        component_3.in_sockets = [None]
        component_3.out_sockets = []

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)
        edge_2 = repository.create_edge(component_2, 0, component_3, 0)

        meta_component = graph.contract(edge_1)

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(len(graph.vertices), 2)
        self.assertIn(meta_component, graph.vertices)
        self.assertIn(component_3, graph.vertices)
        self.assertEqual(edge_2.source, meta_component)
        self.assertEqual(len(meta_component.out_sockets), 1)
        self.assertEqual(meta_component.out_sockets[0][0], edge_2)

    """
    s -> t -> (contract) u
    """
    def test_contract_edge_with_prior_graph(self):
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
        component_3 = component_repository.create(specifications)

        component_1.in_sockets = []
        component_1.out_sockets = [[]]

        component_2.in_sockets = [None]
        component_2.out_sockets = [[]]

        component_3.in_sockets = [None]
        component_3.out_sockets = []

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)
        edge_2 = repository.create_edge(component_2, 0, component_3, 0)

        meta_component = graph.contract(edge_2)

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(len(graph.vertices), 2)
        self.assertIn(meta_component, graph.vertices)
        self.assertIn(component_1, graph.vertices)
        self.assertEqual(edge_1.target, meta_component)
        self.assertEqual(len(meta_component.in_sockets), 1)
        self.assertEqual(meta_component.in_sockets[0], edge_1)

    """
    s <- t -> (contract) u
    """
    def test_contract_edge_with_subsequent_graph_on_source(self):
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
        component_3 = component_repository.create(specifications)

        component_1.in_sockets = []
        component_1.out_sockets = [[], []]

        component_2.in_sockets = [None]
        component_2.out_sockets = []

        component_3.in_sockets = [None]
        component_3.out_sockets = []

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)
        edge_2 = repository.create_edge(component_1, 1, component_3, 0)

        meta_component = graph.contract(edge_2)

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(len(graph.vertices), 2)
        self.assertIn(meta_component, graph.vertices)
        self.assertIn(component_2, graph.vertices)
        self.assertEqual(edge_1.source, meta_component)
        self.assertEqual(len(meta_component.out_sockets), 1)
        self.assertEqual(meta_component.out_sockets[0][0], edge_1)

    """
    s -> (contract) t <- u
    """
    def test_contract_edge_with_prior_graph_on_target(self):
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
        component_3 = component_repository.create(specifications)

        component_1.in_sockets = []
        component_1.out_sockets = [[]]

        component_2.in_sockets = [None, None]
        component_2.out_sockets = []

        component_3.in_sockets = []
        component_3.out_sockets = [[]]

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)
        edge_2 = repository.create_edge(component_3, 0, component_2, 1)

        meta_component = graph.contract(edge_2)

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(len(graph.vertices), 2)
        self.assertIn(meta_component, graph.vertices)
        self.assertIn(component_1, graph.vertices)
        self.assertEqual(edge_1.target, meta_component)
        self.assertEqual(len(meta_component.in_sockets), 1)
        self.assertEqual(meta_component.in_sockets[0], edge_1)

    """
    s -> (contract) t <- s
    """

    def test_contract_edge_with_multiple_edges(self):
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

        component_1.in_sockets = []
        component_1.out_sockets = [[], []]

        component_2.in_sockets = [None, None]
        component_2.out_sockets = []

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)
        edge_2 = repository.create_edge(component_1, 1, component_2, 1)

        meta_component = graph.contract(edge_1)

        self.assertEqual(len(graph.edges), 0)
        self.assertEqual(len(graph.vertices), 1)
        self.assertIn(meta_component, graph.vertices)

    """
    s -> (contract) t -> u <- s
    """

    def test_contract_edge_triangle(self):
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
        component_3 = component_repository.create(specifications)

        component_1.in_sockets = []
        component_1.out_sockets = [[]]

        component_2.in_sockets = [None]
        component_2.out_sockets = [[]]

        component_3.in_sockets = [None, None]
        component_3.out_sockets = []

        edge_1 = repository.create_edge(component_1, 0, component_2, 0)
        edge_2 = repository.create_edge(component_2, 0, component_3, 0)
        edge_3 = repository.create_edge(component_1, 0, component_3, 1)

        meta_component = graph.contract(edge_1)

        self.assertEqual(len(graph.edges), 2)
        self.assertEqual(len(graph.vertices), 2)
        self.assertIn(meta_component, graph.vertices)

        self.assertEqual(len(meta_component.out_sockets), 2)
        self.assertIn(meta_component.out_sockets[0][0], graph.edges)
        self.assertIn(meta_component.out_sockets[1][0], graph.edges)
        self.assertIn(component_3.in_sockets[0], graph.edges)
        self.assertIn(component_3.in_sockets[1], graph.edges)