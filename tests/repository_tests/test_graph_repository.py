import unittest

from tests.setup_holder import SetupHolder

from Mindblocks.model.creation_component.creation_component_in_socket import CreationComponentInSocket
from Mindblocks.model.creation_component.creation_component_out_socket import CreationComponentOutSocket
from Mindblocks.repository.canvas_repository.canvas_repository import CanvasRepository
from Mindblocks.repository.component_type_repository.component_type_repository import ComponentTypeRepository
from Mindblocks.repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from Mindblocks.repository.graph_repository.graph_repository import GraphRepository
from Mindblocks.repository.graph_repository.graph_specifications import GraphSpecifications
from Mindblocks.repository.identifier_repository.identifier_repository import IdentifierRepository


class TestGraphRepository(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testCreateAssignsUID(self):

        specs = GraphSpecifications()

        element_1 = self.setup_holder.graph_repository.create(specs)
        element_2 = self.setup_holder.graph_repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):

        specs = GraphSpecifications()

        self.assertEqual(0, self.setup_holder.graph_repository.count())

        element_1 = self.setup_holder.graph_repository.create(specs)

        self.assertEqual(1, self.setup_holder.graph_repository.count())

        element_2 = self.setup_holder.graph_repository.create(specs)

        self.assertEqual(2, self.setup_holder.graph_repository.count())

    def testMergeGraphsCombinesVertices(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.setup_holder.component_repository.create(c_spec)
        c2 = self.setup_holder.component_repository.create(c_spec)

        g1 = c1.graph
        g2 = c2.graph

        self.assertNotEqual(g1, g2)

        self.setup_holder.graph_repository.merge(g1, g2)

        self.assertIn(c1, g1.get_vertices())
        self.assertIn(c2, g1.get_vertices())

        self.assertEqual(g1, c1.graph)
        self.assertEqual(g1, c2.graph)

    def testMergeGraphsRemovesSecond(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.setup_holder.component_repository.create(c_spec)
        c2 = self.setup_holder.component_repository.create(c_spec)

        g1 = c1.graph
        g2 = c2.graph

        self.assertNotEqual(g1, g2)

        self.setup_holder.graph_repository.merge(g1, g2)

        self.assertEqual(1, self.setup_holder.graph_repository.count())

        g_retrieved = self.setup_holder.graph_repository.get(GraphSpecifications())[0]
        self.assertEqual(g1.identifier, g_retrieved.identifier)
        self.assertEqual(g1, g_retrieved)

    def testCanAddEdgeWithinGraph(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.setup_holder.component_repository.create(c_spec)
        c2 = self.setup_holder.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        g1 = c1.graph
        g2 = c2.graph

        self.setup_holder.graph_repository.merge(g1, g2)
        self.setup_holder.graph_repository.add_edge(out_s, in_s)

        self.assertEqual(1, len(g1.edges))

        edge = g1.edges[0]

        self.assertEqual(out_s, edge.source_socket)
        self.assertEqual(in_s, edge.target_socket)
        self.assertEqual([edge], out_s.edges)
        self.assertEqual(edge, in_s.edge)

    def testCanAddEdgeBetweenGraphs(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.setup_holder.component_repository.create(c_spec)
        c2 = self.setup_holder.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        self.assertEqual(0, len(c1.graph.edges))
        self.assertEqual(0, len(c2.graph.edges))
        self.assertNotEqual(c1.graph, c2.graph)

        self.setup_holder.graph_repository.add_edge(out_s, in_s)

        self.assertEqual(1, len(c1.graph.edges))
        self.assertEqual(c1.graph, c2.graph)

    def testMergeGraphsCombinesEdges(self):
        c_spec = CreationComponentSpecifications()
        c1 = self.setup_holder.component_repository.create(c_spec)
        c2 = self.setup_holder.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        self.setup_holder.graph_repository.add_edge(out_s, in_s)

        g1 = c1.graph

        self.assertEqual(1, g1.count_edges())

        c_spec = CreationComponentSpecifications()
        c1 = self.setup_holder.component_repository.create(c_spec)
        c2 = self.setup_holder.component_repository.create(c_spec)

        out_s = CreationComponentOutSocket(c1, "test_out")
        in_s = CreationComponentInSocket(c2, "test_in")

        c1.add_out_socket(out_s)
        c2.add_in_socket(in_s)

        self.setup_holder.graph_repository.add_edge(out_s, in_s)

        g2 = c1.graph

        self.assertEqual(1, g2.count_edges())

        self.setup_holder.graph_repository.merge(g1, g2)

        self.assertEqual(2, g1.count_edges())


