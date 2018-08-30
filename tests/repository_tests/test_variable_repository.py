import unittest
from Mindblocks.repository.identifier_repository.identifier_repository import IdentifierRepository
from Mindblocks.repository.variable_repository.variable_repository import VariableRepository
from Mindblocks.repository.variable_repository.variable_specifications import VariableSpecifications


class TestVariableRepository(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.variable_repository = VariableRepository(self.identifier_repository)

    def testCreateAssignsUID(self):

        specs = VariableSpecifications()

        element_1 = self.variable_repository.create(specs)
        element_2 = self.variable_repository.create(specs)

        self.assertIsNotNone(element_1)
        self.assertIsNotNone(element_2)
        self.assertIsNotNone(element_1.identifier)
        self.assertIsNotNone(element_2.identifier)
        self.assertNotEqual(element_1.identifier, element_2.identifier)

    def testCreateAddsToRepository(self):

        specs = VariableSpecifications()

        self.assertEqual(0, self.variable_repository.count())

        element_1 = self.variable_repository.create(specs)

        self.assertEqual(1, self.variable_repository.count())

        element_2 = self.variable_repository.create(specs)

        self.assertEqual(2, self.variable_repository.count())

    def testGetByName(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_retrieved = self.variable_repository.get(specs)

        self.assertEqual(1, len(element_retrieved))
        self.assertEqual(element_1.identifier, element_retrieved[0].identifier)
        self.assertEqual(element_1, element_retrieved[0])

        specs.name = "falseTestName"

        element_retrieved = self.variable_repository.get(specs)
        self.assertEqual(0, len(element_retrieved))

    def testGetValueAllModes(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_1.set_value(45)

        element_retrieved = self.variable_repository.get(specs)[0]

        self.assertEqual(45, element_retrieved.get_value(mode=None))
        self.assertEqual(45, element_retrieved.get_value(mode="train"))
        self.assertEqual(45, element_retrieved.get_value(mode="validate"))
        self.assertEqual(45, element_retrieved.get_value(mode="test"))

    def testGetValueSpecificMode(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_1.set_value(45, mode="validate")
        element_1.set_value(547, mode="test")

        element_retrieved = self.variable_repository.get(specs)[0]

        self.assertEqual(None, element_retrieved.get_value(mode=None))
        self.assertEqual(None, element_retrieved.get_value(mode="train"))
        self.assertEqual(45, element_retrieved.get_value(mode="validate"))
        self.assertEqual(547, element_retrieved.get_value(mode="test"))

    def testGetValueSpecificModeWithDefault(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_1.set_value(233, mode=None)
        element_1.set_value(45, mode="validate")
        element_1.set_value(547, mode="test")

        element_retrieved = self.variable_repository.get(specs)[0]

        self.assertEqual(233, element_retrieved.get_value(mode=None))
        self.assertEqual(233, element_retrieved.get_value(mode="train"))
        self.assertEqual(45, element_retrieved.get_value(mode="validate"))
        self.assertEqual(547, element_retrieved.get_value(mode="test"))