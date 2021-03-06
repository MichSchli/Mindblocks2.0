import unittest

from tests.setup_holder import SetupHolder

from Mindblocks.repository.component_type_repository.component_type_repository import ComponentTypeRepository
from Mindblocks.repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from Mindblocks.repository.identifier_repository.identifier_repository import IdentifierRepository


class TestComponentTypeRepository(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder(load_default_types=False)

    def testCreateAssignsUID(self):
        repository = self.setup_holder.type_repository

        specs = ComponentTypeSpecifications()

        element_1 = repository.create(specs)
        element_2 = repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):
        repository = self.setup_holder.type_repository

        specs = CreationComponentSpecifications()

        self.assertEqual(0, repository.count())

        element_1 = repository.create(specs)

        self.assertEqual(1, repository.count())

        element_2 = repository.create(specs)

        self.assertEqual(2, repository.count())

    def testGetByName(self):
        repository = self.setup_holder.type_repository

        specs = ComponentTypeSpecifications()
        specs.name = "testComponent"

        element_1 = repository.create(specs)
        element_retrieved = repository.get(specs)

        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_1.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_1, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = repository.get(specs)
        self.assertEqual(0, len(element_retrieved))