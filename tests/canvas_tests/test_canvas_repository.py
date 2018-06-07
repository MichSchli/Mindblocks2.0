import unittest

from controller.controller import Controller
from repository.canvas.canvas_repository import CanvasRepository
from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestCanvasRepository(unittest.TestCase):

    def test_add_canvas(self):
        identifier_repository = IdentifierRepository()
        repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 1)

    def test_add_multiple_canvases(self):
        identifier_repository = IdentifierRepository()
        repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertEqual(len(repository.elements), 2)

    def test_add_canvas_with_name(self):
        identifier_repository = IdentifierRepository()
        repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        specifications.name = "test_name"
        repository.create(specifications)

        self.assertEqual(repository.elements[0].get_name(), "test_name")

    def test_add_canvas_identifier(self):
        identifier_repository = IdentifierRepository()
        repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        repository.create(specifications)
        repository.create(specifications)

        self.assertIsNotNone(repository.elements[0].identifier)
        self.assertIsNotNone(repository.elements[1].identifier)
        self.assertNotEqual(repository.elements[0].identifier, repository.elements[1].identifier)

    def test_get_by_name(self):
        identifier_repository = IdentifierRepository()
        repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        repository.create(specifications)
        specifications.name = "test_name"
        canvas_2 = repository.create(specifications)

        canvases = repository.get(specifications)

        self.assertEqual(len(canvases), 1)
        self.assertEqual(canvases[0].name, canvas_2.name)
        self.assertEqual(canvases[0].identifier, canvas_2.identifier)

    def test_get_by_id(self):
        identifier_repository = IdentifierRepository()
        repository = CanvasRepository(identifier_repository)
        specifications = CanvasSpecifications()
        repository.create(specifications)
        canvas_2 = repository.create(specifications)

        specifications = CanvasSpecifications()
        specifications.identifier = canvas_2.identifier

        canvases = repository.get(specifications)

        self.assertEqual(len(canvases), 1)
        self.assertEqual(canvases[0].name, canvas_2.name)
        self.assertEqual(canvases[0].identifier, canvas_2.identifier)