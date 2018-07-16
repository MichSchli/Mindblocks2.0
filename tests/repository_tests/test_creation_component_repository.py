import unittest

from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestCreationComponentRepository(unittest.TestCase):

    def testCreateAssignsUID(self):
        identifier_repository = IdentifierRepository()
        repository = CreationComponentRepository(identifier_repository)

        specs = CreationComponentSpecifications()

        element_1 = repository.create(specs)
        element_2 = repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):
        identifier_repository = IdentifierRepository()
        repository = CreationComponentRepository(identifier_repository)

        specs = CreationComponentSpecifications()

        self.assertEquals(0, repository.count())

        element_1 = repository.create(specs)

        self.assertEquals(1, repository.count())

        element_2 = repository.create(specs)

        self.assertEquals(2, repository.count())

    def testGetByName(self):
        identifier_repository = IdentifierRepository()
        repository = CreationComponentRepository(identifier_repository)

        specs = CreationComponentSpecifications()
        specs.name = "testComponent"

        element_1 = repository.create(specs)
        element_retrieved = repository.get(specs)

        self.assertEquals(1, len(element_retrieved))
        self.assertEquals(element_1.identifier, element_retrieved[0].identifier)
        self.assertEquals(element_1, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = repository.get(specs)
        self.assertEquals(0, len(element_retrieved))