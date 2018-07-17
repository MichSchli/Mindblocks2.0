import unittest

from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestCreationComponentRepository(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.repository = CreationComponentRepository(self.identifier_repository, self.type_repository)

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

        self.assertEquals(0, self.repository.count())

        element_1 = self.repository.create(specs)

        self.assertEquals(1, self.repository.count())

        element_2 = self.repository.create(specs)

        self.assertEquals(2, self.repository.count())

    def testGetByName(self):
        specs = CreationComponentSpecifications()
        specs.name = "testComponent"

        element_1 = self.repository.create(specs)
        element_retrieved = self.repository.get(specs)

        self.assertEquals(1, len(element_retrieved))
        self.assertEquals(element_1.identifier, element_retrieved[0].identifier)
        self.assertEquals(element_1, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = self.repository.get(specs)
        self.assertEquals(0, len(element_retrieved))

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