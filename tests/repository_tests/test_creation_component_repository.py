import unittest

from Mindblocks.repository.canvas_repository.canvas_repository import CanvasRepository
from Mindblocks.repository.canvas_repository.canvas_specifications import CanvasSpecifications
from Mindblocks.repository.component_type_repository.component_type_repository import ComponentTypeRepository
from Mindblocks.repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from Mindblocks.repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from Mindblocks.repository.graph.graph_repository import GraphRepository
from Mindblocks.repository.graph.graph_specifications import GraphSpecifications
from Mindblocks.repository.identifier.identifier_repository import IdentifierRepository


class TestCreationComponentRepository(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.canvas_repository = CanvasRepository(self.identifier_repository)
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.graph_repository = GraphRepository(self.identifier_repository)
        self.repository = CreationComponentRepository(self.identifier_repository,
                                                      self.type_repository,
                                                      self.canvas_repository,
                                                      self.graph_repository)

    def testCreateAssignsUID(self):
        specs = CreationComponentSpecifications()

        element_1 = self.repository.create(specs)
        element_2 = self.repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):

        specs = CreationComponentSpecifications()

        self.assertEqual(0, self.repository.count())

        element_1 = self.repository.create(specs)

        self.assertEqual(1, self.repository.count())

        element_2 = self.repository.create(specs)

        self.assertEqual(2, self.repository.count())

    def testGetByName(self):
        specs = CreationComponentSpecifications()
        specs.name = "testComponent"

        element_1 = self.repository.create(specs)
        element_retrieved = self.repository.get(specs)

        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_1.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_1, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = self.repository.get(specs)
        self.assertEqual(0, len(element_retrieved))

    def testCreateWithComponentTypeName(self):
        type_specs = ComponentTypeSpecifications()
        type_specs.name = "TestType"

        component_type = self.type_repository.create(type_specs)

        specs = CreationComponentSpecifications()
        specs.component_type_name = "TestType"

        element = self.repository.create(specs)

        self.assertIsNotNone(element.component_type)
        self.assertEqual(component_type, element.component_type)
        self.assertEqual("TestType", element.get_component_type_name())

    def testCreateWithComponentTypeAssignsDefaultValue(self):
        type_specs = ComponentTypeSpecifications()
        type_specs.name = "TestType"

        component_type = self.type_repository.create(type_specs)

        def test_assign(dic):
            dic["test_field"] = "test_value"
        component_type.assign_default_value = test_assign

        specs = CreationComponentSpecifications()
        specs.component_type_name = "TestType"

        element = self.repository.create(specs)

        self.assertIsNotNone(element.component_value)
        self.assertEqual({"test_field": "test_value"}, element.component_value)

    def testCreateWithCanvasName(self):
        canvas_specs = CanvasSpecifications()
        canvas_specs.name = "TestCanvas"

        canvas = self.canvas_repository.create(canvas_specs)

        specs = CreationComponentSpecifications()
        specs.canvas_name = "TestCanvas"

        element = self.repository.create(specs)

        self.assertIsNotNone(element.canvas)
        self.assertEqual(canvas, element.canvas)
        self.assertEqual(1, len(canvas.components))
        self.assertEqual(element, canvas.components[0])
        self.assertEqual("TestCanvas", element.get_canvas_name())

    def testCreateAddsToGraphRepository(self):
        specs = CreationComponentSpecifications()

        self.assertEqual(0, self.graph_repository.count())

        element_1 = self.repository.create(specs)

        self.assertEqual(1, self.graph_repository.count())

        specs = GraphSpecifications()
        graph = self.graph_repository.get(specs)[0]

        self.assertEqual(element_1.get_graph_identifier(), graph.identifier)
        self.assertEqual(element_1.graph, graph)
        self.assertEqual(1, graph.count_vertices())
        self.assertEqual(element_1, graph.get_vertices()[0])

    def testCreateAssignsOutSockets(self):
        type_specs = ComponentTypeSpecifications()
        type_specs.name = "TestType"

        component_type = self.type_repository.create(type_specs)
        component_type.out_sockets = ["out_1", "out_2"]

        specs = CreationComponentSpecifications()
        specs.component_type_name = "TestType"

        element = self.repository.create(specs)

        self.assertIsNotNone(element.out_sockets)
        self.assertIsNotNone(element.in_sockets)
        self.assertEqual(2, element.count_out_sockets())
        self.assertEqual(0, element.count_in_sockets())

        self.assertIn("out_1", element.out_sockets)
        self.assertIn("out_2", element.out_sockets)
        self.assertIsNotNone(element.get_out_socket("out_1"))
        self.assertIsNotNone(element.get_out_socket("out_2"))

    def testCreateAssignsInSockets(self):
        type_specs = ComponentTypeSpecifications()
        type_specs.name = "TestType"

        component_type = self.type_repository.create(type_specs)
        component_type.in_sockets = ["in_1", "in_2"]

        specs = CreationComponentSpecifications()
        specs.component_type_name = "TestType"

        element = self.repository.create(specs)

        self.assertIsNotNone(element.out_sockets)
        self.assertIsNotNone(element.in_sockets)
        self.assertEqual(0, element.count_out_sockets())
        self.assertEqual(2, element.count_in_sockets())

        self.assertIn("in_1", element.in_sockets)
        self.assertIn("in_1", element.in_sockets)
        self.assertIsNotNone(element.get_in_socket("in_1"))
        self.assertIsNotNone(element.get_in_socket("in_2"))

    def testCreateAssignsLanguage(self):
        type_specs = ComponentTypeSpecifications()
        type_specs.name = "TestType"

        component_type = self.type_repository.create(type_specs)
        component_type.languages = ["test_language", "other_test_language"]

        specs = CreationComponentSpecifications()
        specs.component_type_name = "TestType"

        element = self.repository.create(specs)

        self.assertIsNotNone(element.language)
        self.assertEqual("test_language", element.language)