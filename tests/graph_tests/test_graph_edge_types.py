import unittest

from model.component.component_model import ComponentModel
from model.graph.edge_model import Edge
from model.graph.value_type_model import ValueTypeModel
from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestGraphEdgeTypes(unittest.TestCase):

    def test_source_type_used(self):
        c1 = ComponentModel()
        c2 = ComponentModel()
        c1.out_value_types = [ValueTypeModel("float", 1)]
        edge = Edge(c1, c2)
        edge.source_socket = 0

        self.assertIsNotNone(edge.get_value_type())
        self.assertNotEqual(edge.get_value_type(), c1.out_value_types[0])
        self.assertEqual(edge.get_value_type().type, "float")

    def test_source_dims_used(self):
        c1 = ComponentModel()
        c2 = ComponentModel()
        c1.out_value_types = [ValueTypeModel("int", 1)]
        edge = Edge(c1, c2)
        edge.source_socket = 0

        self.assertIsNotNone(edge.get_value_type())
        self.assertNotEqual(edge.get_value_type(), c1.out_value_types[0])
        self.assertEqual(edge.get_value_type().dim, 1)

    def test_cast_type_used(self):
        c1 = ComponentModel()
        c2 = ComponentModel()
        c1.out_value_types = [ValueTypeModel("float", 1)]
        edge = Edge(c1, c2)
        edge.cast_to = "int"
        edge.source_socket = 0

        self.assertIsNotNone(edge.get_value_type())
        self.assertNotEqual(edge.get_value_type(), c1.out_value_types[0])
        self.assertEqual(edge.get_value_type().type, "int")

    def test_cast_type_source_dims_used(self):
        c1 = ComponentModel()
        c2 = ComponentModel()
        c1.out_value_types = [ValueTypeModel("int", 1)]
        edge = Edge(c1, c2)
        edge.cast_to = "int"
        edge.source_socket = 0

        self.assertIsNotNone(edge.get_value_type())
        self.assertNotEqual(edge.get_value_type(), c1.out_value_types[0])
        self.assertEqual(edge.get_value_type().dim, 1)