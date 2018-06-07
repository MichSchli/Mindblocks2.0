import unittest

from model.component.component_model import ComponentModel
from repository.canvas.canvas_repository import CanvasRepository
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestGraphRepository(unittest.TestCase):

    def test_add(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 1)

    def test_add_multiple(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 2)

    def test_add_with_name(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        specifications.name = "test_name"
        repository.create(specifications)

        self.assertEqual(repository.elements[0].get_name(), "test_name")

    def test_add_identifiers(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertIsNotNone(repository.elements[0].identifier)
        self.assertIsNotNone(repository.elements[1].identifier)
        self.assertNotEqual(repository.elements[0].identifier, repository.elements[1].identifier)

    def test_get_by_name(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        repository.create(specifications)
        specifications.name = "test_name"
        element_2 = repository.create(specifications)

        elements = repository.get(specifications)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].name, element_2.name)
        self.assertEqual(elements[0].identifier, element_2.identifier)

    def test_get_by_id(self):
        identifier_repository = IdentifierRepository()
        repository = GraphRepository(identifier_repository, None)
        specifications = GraphSpecifications()
        repository.create(specifications)
        element_2 = repository.create(specifications)

        specifications = GraphSpecifications()
        specifications.identifier = element_2.identifier

        elements = repository.get(specifications)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].name, element_2.name)
        self.assertEqual(elements[0].identifier, element_2.identifier)

    def test_add_to_canvas_with_id(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        canvas = canvas_repository.create(specifications)
        self.assertEqual(len(canvas.get_components()), 0)
        canvas_id = canvas.identifier

        repository = GraphRepository(identifier_repository, canvas_repository)
        specifications = GraphSpecifications()
        specifications.canvas_id = canvas_id
        element = repository.create(specifications)

        self.assertEqual(element.canvas_id, canvas_id)
        self.assertEqual(len(canvas.get_graphs()), 1)
        self.assertEqual(canvas.get_graphs()[0].identifier, element.identifier)

    def test_add_to_canvas_with_name(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        specifications.name = "test_name"
        canvas = canvas_repository.create(specifications)
        self.assertEqual(len(canvas.get_components()), 0)
        canvas_id = canvas.identifier

        repository = GraphRepository(identifier_repository, canvas_repository)
        specifications = GraphSpecifications()
        specifications.canvas_name = "test_name"
        element = repository.create(specifications)

        self.assertEqual(element.canvas_id, canvas_id)
        self.assertEqual(element.canvas_name, "test_name")
        self.assertEqual(len(canvas.get_graphs()), 1)
        self.assertEqual(canvas.get_graphs()[0].identifier, element.identifier)

    def test_join_only_left_in_repository(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        repository = GraphRepository(identifier_repository, canvas_repository)
        specifications = GraphSpecifications()
        graph_1 = repository.create(specifications)
        graph_2 = repository.create(specifications)

        self.assertEqual(len(repository.elements), 2)

        repository.join_graphs(graph_1, graph_2)

        self.assertEqual(len(repository.elements), 1)
        self.assertEqual(repository.elements[0], graph_1)

    def test_join_left_has_all_components(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        repository = GraphRepository(identifier_repository, canvas_repository)
        specifications = GraphSpecifications()
        graph_1 = repository.create(specifications)
        graph_2 = repository.create(specifications)

        component_1 = ComponentModel()
        component_1.graph_id = graph_1.identifier
        graph_1.add_component(component_1)

        component_2 = ComponentModel()
        component_2.graph_id = graph_2.identifier
        graph_2.add_component(component_2)

        repository.join_graphs(graph_1, graph_2)

        self.assertEqual(len(graph_1.vertices), 2)
        self.assertEqual(graph_1.vertices[0], component_1)
        self.assertEqual(graph_1.vertices[1], component_2)

    def test_join_components_all_use_left_id(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        repository = GraphRepository(identifier_repository, canvas_repository)
        specifications = GraphSpecifications()
        graph_1 = repository.create(specifications)
        graph_2 = repository.create(specifications)

        component_1 = ComponentModel()
        component_1.graph_id = graph_1.identifier
        graph_1.add_component(component_1)

        component_2 = ComponentModel()
        component_2.graph_id = graph_2.identifier
        graph_2.add_component(component_2)

        repository.join_graphs(graph_1, graph_2)

        self.assertEqual(graph_1.identifier, component_1.graph_id)
        self.assertEqual(graph_1.identifier, component_2.graph_id)

    def test_join_removed_from_canvas(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        canvas = canvas_repository.create(specifications)
        self.assertEqual(len(canvas.get_components()), 0)
        canvas_id = canvas.identifier

        repository = GraphRepository(identifier_repository, canvas_repository)
        specifications = GraphSpecifications()
        specifications.canvas_id = canvas_id
        graph_1 = repository.create(specifications)
        graph_2 = repository.create(specifications)

        self.assertEqual(len(canvas.graphs), 2)

        repository.join_graphs(graph_1, graph_2)

        self.assertEqual(len(canvas.graphs), 1)
        self.assertEqual(canvas.graphs[0], graph_1)