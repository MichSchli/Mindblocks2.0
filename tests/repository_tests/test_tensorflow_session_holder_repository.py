import unittest

from Mindblocks.repository.canvas_repository.canvas_repository import CanvasRepository
from Mindblocks.repository.canvas_repository.canvas_specifications import CanvasSpecifications
from Mindblocks.repository.identifier_repository.identifier_repository import IdentifierRepository
from Mindblocks.repository.tensorflow_session_repository.tensorflow_session_specifications import \
    TensorflowSessionSpecifications
from tests.setup_holder import SetupHolder


class TestTensorflowSessionRepository(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()
        self.repository = self.setup_holder.tensorflow_session_repository

    def testCreateAssignsUID(self):
        specs = TensorflowSessionSpecifications()

        element_1 = self.repository.create(specs)
        element_2 = self.repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testNewAssignsUID(self):
        element_1 = self.repository.new()
        element_2 = self.repository.new()

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAssignsTfSession(self):

        specs = TensorflowSessionSpecifications()

        element_1 = self.repository.create(specs)
        element_2 = self.repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.get_session())
        self.assertIsNotNone(element_2.get_session())
        self.assertNotEqual(element_1.get_session(), element_2.get_session())

    def testCreateAddsToRepository(self):

        specs = CanvasSpecifications()

        self.assertEqual(0, self.repository.count())

        element_1 = self.repository.create(specs)

        self.assertEqual(1, self.repository.count())

        element_2 = self.repository.create(specs)

        self.assertEqual(2, self.repository.count())

    def testGetById(self):
        specs = TensorflowSessionSpecifications()

        element_1 = self.repository.create(specs)
        id = element_1.identifier
        specs.identifier = id

        element_retrieved = self.repository.get(specs)

        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_1.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_1.get_session(), element_retrieved[0].get_session())
        self.assertEqual(element_1, element_retrieved[0])

        specs.identifier = self.setup_holder.identifier_repository.create()

        element_retrieved = self.repository.get(specs)
        self.assertEqual(0, len(element_retrieved))