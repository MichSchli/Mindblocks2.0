import unittest

from tests.setup_holder import SetupHolder

from Mindblocks.repository.canvas_repository.canvas_repository import CanvasRepository
from Mindblocks.repository.canvas_repository.canvas_specifications import CanvasSpecifications
from Mindblocks.repository.identifier_repository.identifier_repository import IdentifierRepository


class TestCanvasRepository(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()
        self.repository = self.setup_holder.canvas_repository

    def testCreateAssignsUID(self):

        specs = CanvasSpecifications()

        element_1 = self.repository.create(specs)
        element_2 = self.repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):

        specs = CanvasSpecifications()

        self.assertEqual(0, self.repository.count())

        element_1 = self.repository.create(specs)

        self.assertEqual(1, self.repository.count())

        element_2 = self.repository.create(specs)

        self.assertEqual(2, self.repository.count())

    def testGetByName(self):
        specs = CanvasSpecifications()
        specs.name = "testElement"

        element_1 = self.repository.create(specs)
        element_retrieved = self.repository.get(specs)

        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_1.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_1, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = self.repository.get(specs)
        self.assertEqual(0, len(element_retrieved))

    def testCanvasComponentsIsEmptyList(self):
        specs = CanvasSpecifications()
        canvas = self.repository.create(specs)
        self.assertEqual([], canvas.get_components())