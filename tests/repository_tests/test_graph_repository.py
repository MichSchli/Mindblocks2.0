import unittest

from model.creation_component.creation_component_in_socket import CreationComponentInSocket
from model.creation_component.creation_component_out_socket import CreationComponentOutSocket
from repository.canvas_repository.canvas_repository import CanvasRepository
from repository.canvas_repository.canvas_specifications import CanvasSpecifications
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestGraphRepository(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.canvas_repository = CanvasRepository(self.identifier_repository)
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.graph_repository = GraphRepository(self.identifier_repository)
        self.component_repository = CreationComponentRepository(self.identifier_repository,
                                                      self.type_repository,
                                                      self.canvas_repository,
                                                      self.graph_repository)

    def testCreateAssignsUID(self):

        specs = GraphSpecifications()

        element_1 = self.graph_repository.create(specs)
        element_2 = self.graph_repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):

        specs = GraphSpecifications()

        self.assertEquals(0, self.graph_repository.count())

        element_1 = self.graph_repository.create(specs)

        self.assertEquals(1, self.graph_repository.count())

        element_2 = self.graph_repository.create(specs)

        self.assertEquals(2, self.graph_repository.count())

    def testMergeGraphsCombinesVertices(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.component_repository.create(c_spec)
        c2 = self.component_repository.create(c_spec)

        g1 = c1.graph
        g2 = c2.graph

        self.assertNotEqual(g1, g2)

        self.graph_repository.merge(g1, g2)

        self.assertIn(c1, g1.get_vertices())
        self.assertIn(c2, g1.get_vertices())

        self.assertEquals(g1, c1.graph)
        self.assertEquals(g1, c2.graph)

    def testMergeGraphsRemovesSecond(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.component_repository.create(c_spec)
        c2 = self.component_repository.create(c_spec)

        g1 = c1.graph
        g2 = c2.graph

        self.assertNotEqual(g1, g2)

        self.graph_repository.merge(g1, g2)

        self.assertEquals(1, self.graph_repository.count())

        g_retrieved = self.graph_repository.get(GraphSpecifications())[0]
        self.assertEquals(g1.identifier, g_retrieved.identifier)
        self.assertEquals(g1, g_retrieved)

    def testCanAddEdgeWithinGraph(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.component_repository.create(c_spec)
        c2 = self.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        g1 = c1.graph
        g2 = c2.graph

        self.graph_repository.merge(g1, g2)
        self.graph_repository.add_edge(out_s, in_s)

        self.assertEquals(1, len(g1.edges))

        edge = g1.edges[0]

        self.assertEquals(out_s, edge.source_socket)
        self.assertEquals(in_s, edge.target_socket)
        self.assertEquals([edge], out_s.edges)
        self.assertEquals(edge, in_s.edge)

    def testCanAddEdgeBetweenGraphs(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.component_repository.create(c_spec)
        c2 = self.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        self.assertEquals(0, len(c1.graph.edges))
        self.assertEquals(0, len(c2.graph.edges))
        self.assertNotEqual(c1.graph, c2.graph)

        self.graph_repository.add_edge(out_s, in_s)

        self.assertEquals(1, len(c1.graph.edges))
        self.assertEquals(c1.graph, c2.graph)

    def testMergeGraphsCombinesEdges(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.component_repository.create(c_spec)
        c2 = self.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        self.graph_repository.add_edge(out_s, in_s)

        g1 = c1.graph

        self.assertEquals(1, g1.count_edges())

        c_spec = CreationComponentSpecifications()
        c1 = self.component_repository.create(c_spec)
        c2 = self.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        self.graph_repository.add_edge(out_s, in_s)

        g2 = c1.graph

        self.assertEquals(1, g2.count_edges())

        self.graph_repository.merge(g1, g2)

        self.assertEquals(2, g1.count_edges())


