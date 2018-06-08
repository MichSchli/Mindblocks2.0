import unittest

from controller.controller import Controller
from model.component.component_type.component_type_model import ComponentTypeModel
from repository.canvas.canvas_repository import CanvasRepository
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.component.component_type.component_type_repository import ComponentTypeRepository
from repository.component.component_type.component_type_specifications import ComponentTypeSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestComponentTypeRepository(unittest.TestCase):

    def test_create(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)
        specifications = ComponentTypeSpecifications()
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 1)

    def test_create_multiple(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)
        specifications = ComponentTypeSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 2)

    def test_create_with_name(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)
        specifications = ComponentTypeSpecifications()
        specifications.name = "test_name"
        repository.create(specifications)

        self.assertEqual(repository.elements[0].get_name(), "test_name")

    def test_create_identifiers(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)
        specifications = ComponentTypeSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertIsNotNone(repository.elements[0].identifier)
        self.assertIsNotNone(repository.elements[1].identifier)
        self.assertNotEqual(repository.elements[0].identifier, repository.elements[1].identifier)

    def test_get_by_name(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)
        specifications = ComponentTypeSpecifications()
        repository.create(specifications)
        specifications.name = "test_name"
        element_2 = repository.create(specifications)

        elements = repository.get(specifications)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].name, element_2.name)
        self.assertEqual(elements[0].identifier, element_2.identifier)

    def test_get_by_id(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)
        specifications = ComponentTypeSpecifications()
        repository.create(specifications)
        element_2 = repository.create(specifications)

        specifications = ComponentTypeSpecifications()
        specifications.identifier = element_2.identifier

        elements = repository.get(specifications)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].name, element_2.name)
        self.assertEqual(elements[0].identifier, element_2.identifier)

    def test_add_existing(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)

        component_type = ComponentTypeModel()
        repository.add(component_type)

        self.assertEqual(len(repository.elements), 1)
        self.assertEqual(repository.elements[0], component_type)

    def test_add_existing_assigns_id(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)

        component_type = ComponentTypeModel()

        self.assertIsNone(component_type.identifier)

        repository.add(component_type)

        self.assertIsNotNone(component_type.identifier)

        self.assertEqual(len(repository.elements), 1)
        self.assertEqual(repository.elements[0], component_type)
        self.assertEqual(repository.elements[0].identifier, component_type.identifier)

    def test_add_and_get_by_id(self):
        identifier_repository = IdentifierRepository()
        repository = ComponentTypeRepository(identifier_repository)
        component_type = ComponentTypeModel()
        repository.add(component_type)
        id = component_type.identifier

        specifications = ComponentTypeSpecifications()
        specifications.identifier = id
        elements = repository.get(specifications)

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0], component_type)
        self.assertEqual(elements[0].identifier, component_type.identifier)