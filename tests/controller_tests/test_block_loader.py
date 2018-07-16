import unittest

from controller.block_loader.component_loader import ComponentLoader
from helpers.xml.xml_helper import XmlHelper
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestBlockLoader(unittest.TestCase):

    def testLoadsSimpleComponent(self):
        xml_helper = XmlHelper()
        identifier_repository = IdentifierRepository()
        component_repository = CreationComponentRepository(identifier_repository)
        component_loader = ComponentLoader(xml_helper, component_repository)

        text = """<component name="constant_1" type="Constant">
                       <value type="float">5.17</value>
                  </component>"""

        component_loader.load_component(text, 0)

        self.assertEquals(1, component_repository.count())

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"

        components = component_repository.get(spec)

        self.assertEquals(1, len(components))
        self.assertEquals("constant_1", components[0].name)
        self.assertEquals("Constant", components[0].get_component_type_name()())
        self.assertEquals("5.17", components[0].component_value.value)
        self.assertEquals("float", components[0].component_value.value_type)

