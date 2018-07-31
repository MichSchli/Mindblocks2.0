import unittest

from repository.canvas_repository.canvas_repository import CanvasRepository
from repository.canvas_repository.canvas_specifications import CanvasSpecifications
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.identifier.identifier_repository import IdentifierRepository
from repository.variable_repository.variable_repository import VariableRepository
from repository.variable_repository.variable_specifications import VariableSpecifications


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

        element_retrieved = self.repository.get(specs)
        self.assertEqual(0, len(element_retrieved))

    def testGetValueAllModes(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_1.set_value(45)

        element_retrieved = self.variable_repository.get(specs)[0]

        self.assertEquals(45, element_retrieved.get_value(mode=None))
        self.assertEquals(45, element_retrieved.get_value(mode="train"))
        self.assertEquals(45, element_retrieved.get_value(mode="dev"))
        self.assertEquals(45, element_retrieved.get_value(mode="test"))

    def testGetValueSpecificMode(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_1.set_value(45, mode="dev")
        element_1.set_value(547, mode="test")

        element_retrieved = self.variable_repository.get(specs)[0]

        self.assertEquals(None, element_retrieved.get_value(mode=None))
        self.assertEquals(None, element_retrieved.get_value(mode="train"))
        self.assertEquals(45, element_retrieved.get_value(mode="dev"))
        self.assertEquals(547, element_retrieved.get_value(mode="test"))

    def testGetValueSpecificModeWithDefault(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_1.set_value(233, mode=None)
        element_1.set_value(45, mode="dev")
        element_1.set_value(547, mode="test")

        element_retrieved = self.variable_repository.get(specs)[0]

        self.assertEquals(233, element_retrieved.get_value(mode=None))
        self.assertEquals(233, element_retrieved.get_value(mode="train"))
        self.assertEquals(45, element_retrieved.get_value(mode="dev"))
        self.assertEquals(547, element_retrieved.get_value(mode="test"))

    def testReplaceInString(self):
        specs = VariableSpecifications()
        specs.name = "testElement"

        element_1 = self.variable_repository.create(specs)
        element_1.set_value("test value", mode=None)

        source_str = "will this $testElement have the correct value"
        target_str = element_1.replace_in_string(source_str)

        self.assertEquals("will this test value have the correct value", target_str)