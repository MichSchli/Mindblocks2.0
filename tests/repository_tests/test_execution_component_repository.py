import unittest

from Mindblocks.repository.canvas_repository.canvas_repository import CanvasRepository
from Mindblocks.repository.canvas_repository.canvas_specifications import CanvasSpecifications
from Mindblocks.repository.execution_component_repository.execution_component_specifications import \
    ExecutionComponentSpecifications
from Mindblocks.repository.identifier.identifier_repository import IdentifierRepository
from tests.setup_holder import SetupHolder


class TestExecutionComponentRepository(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()
        self.repository = self.setup_holder.execution_component_repository

    def testCreateAssignsUID(self):

        specs = ExecutionComponentSpecifications()

        element_1 = self.repository.create(specs)
        element_2 = self.repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):

        specs = ExecutionComponentSpecifications()

        self.assertEqual(0, self.repository.count())

        element_1 = self.repository.create(specs)

        self.assertEqual(1, self.repository.count())

        element_2 = self.repository.create(specs)

        self.assertEqual(2, self.repository.count())

    def testGetByName(self):
        specs = ExecutionComponentSpecifications()
        specs.name = "testElement"

        element_1 = self.repository.create(specs)
        element_retrieved = self.repository.get(specs)

        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_1.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_1, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = self.repository.get(specs)
        self.assertEqual(0, len(element_retrieved))

    def testGetByNameAndMode(self):
        specs = ExecutionComponentSpecifications()
        specs.name = "testElement"
        specs.mode = "test"

        element_1 = self.repository.create(specs)
        specs.name = "testElement"
        specs.mode = "train"

        element_2 = self.repository.create(specs)

        specs.mode = None
        element_retrieved = self.repository.get(specs)
        self.assertEqual(2, len(element_retrieved))

        specs.mode = "test"
        element_retrieved = self.repository.get(specs)
        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_1.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_1, element_retrieved[0])

        specs.mode = "train"
        element_retrieved = self.repository.get(specs)
        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_2.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_2, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = self.repository.get(specs)
        self.assertEqual(0, len(element_retrieved))