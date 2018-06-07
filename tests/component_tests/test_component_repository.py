import unittest

from repository.canvas.canvas_repository import CanvasRepository
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.component.component_repository import ComponentRepository
from repository.component.component_specifications import ComponentSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.component.component_type.component_type_specifications import ComponentTypeSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.graph.graph_specifications import GraphSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestComponentRepository(unittest.TestCase):

    def test_add(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 1)

    def test_add_multiple(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 2)

    def test_add_with_name(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.name = "test_name"
        repository.create(specifications)

        self.assertEqual(repository.elements[0].get_name(), "test_name")

    def test_add_identifiers(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertIsNotNone(repository.elements[0].identifier)
        self.assertIsNotNone(repository.elements[1].identifier)
        self.assertNotEqual(repository.elements[0].identifier, repository.elements[1].identifier)

    def test_get_by_name(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        repository.create(specifications)
        specifications.name = "test_name"
        element_2 = repository.create(specifications)

        elements = repository.get(specifications)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].name, element_2.name)
        self.assertEqual(elements[0].identifier, element_2.identifier)

    def test_get_by_id(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        repository.create(specifications)
        element_2 = repository.create(specifications)

        specifications = ComponentSpecifications()
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

        graph_repository = GraphRepository(identifier_repository, canvas_repository)
        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, canvas_repository, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.canvas_id = canvas_id
        element = repository.create(specifications)

        self.assertEqual(element.canvas_id, canvas_id)
        self.assertEqual(len(canvas.get_components()), 1)
        self.assertEqual(canvas.get_components()[0].identifier, element.identifier)

    def test_add_to_canvas_with_name(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        specifications.name = "test_name"
        canvas = canvas_repository.create(specifications)
        self.assertEqual(len(canvas.get_components()), 0)
        canvas_id = canvas.identifier

        graph_repository = GraphRepository(identifier_repository, canvas_repository)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, canvas_repository, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.canvas_name = "test_name"
        element = repository.create(specifications)

        self.assertEqual(element.canvas_id, canvas_id)
        self.assertEqual(element.canvas_name, "test_name")
        self.assertEqual(len(canvas.get_components()), 1)
        self.assertEqual(canvas.get_components()[0].identifier, element.identifier)

    def test_add_creates_graph(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        component = repository.create(specifications)

        self.assertIsNotNone(component.graph_id)

    def test_add_multiple_creates_different_graphs(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)
        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        component_1 = repository.create(specifications)
        component_2 = repository.create(specifications)

        self.assertNotEqual(component_1.graph_id, component_2.graph_id)

    def test_add_created_graph_in_repository(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)
        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        component = repository.create(specifications)

        graph_identifier = component.graph_id
        graph_specifications = GraphSpecifications()
        graph_specifications.identifier = graph_identifier
        graphs = graph_repository.get(graph_specifications)

        self.assertEqual(len(graphs), 1)

    def test_add_to_canvas_graph_same_canvas(self):
        identifier_repository = IdentifierRepository()
        canvas_repository = CanvasRepository(identifier_repository)
        graph_repository = GraphRepository(identifier_repository, canvas_repository)
        specifications = CanvasSpecifications()
        canvas = canvas_repository.create(specifications)
        self.assertEqual(len(canvas.get_components()), 0)
        canvas_id = canvas.identifier

        component_type_repository = ComponentTypeRepository(identifier_repository)
        repository = ComponentRepository(identifier_repository, canvas_repository, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.canvas_id = canvas_id
        element = repository.create(specifications)

        graph_identifier = element.graph_id
        graph_specifications = GraphSpecifications()
        graph_specifications.identifier = graph_identifier
        graph = graph_repository.get(graph_specifications)[0]

        self.assertEqual(element.canvas_id, graph.canvas_id)
        self.assertEqual(len(canvas.get_graphs()), 1)
        self.assertEqual(canvas.get_graphs()[0].identifier, element.graph_id)

    def test_creates_with_component_type(self):
        identifier_repository = IdentifierRepository()
        graph_repository = GraphRepository(identifier_repository, None)
        component_type_repository = ComponentTypeRepository(identifier_repository)
        specs = ComponentTypeSpecifications()
        specs.name = "test_type"
        component_type = component_type_repository.create(specs)

        repository = ComponentRepository(identifier_repository, None, graph_repository, component_type_repository)
        specifications = ComponentSpecifications()
        specifications.component_type_name = specs.name
        component = repository.create(specifications)

        self.assertIsNotNone(component.component_type)
        self.assertEqual(component_type.identifier, component.component_type.identifier)