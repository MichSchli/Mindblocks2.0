import unittest

from controller.block_loader.component_loader import ComponentLoader
from helpers.xml.xml_helper import XmlHelper
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.identifier.identifier_repository import IdentifierRepository


class TestBlockLoader(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.component_repository = CreationComponentRepository(self.identifier_repository, self.type_repository)

        self.xml_helper = XmlHelper()
        self.component_loader = ComponentLoader(self.xml_helper, self.component_repository)

    def testLoadsSimpleComponent(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.type_repository.create(component_type_spec)

        def test_assign(dic):
            dic["value"] = "test_value"
            dic["value_type"] = "test_value"

        component_type.assign_default_value = test_assign

        text = """<component name="constant_1" type="Constant"><value>5.17</value><type>float</type></component>"""

        self.component_loader.load_component(text, 0)

        self.assertEquals(1, self.component_repository.count())

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"

        components = self.component_repository.get(spec)

        self.assertEquals(1, len(components))
        self.assertEquals("constant_1", components[0].name)
        self.assertEquals("Constant", components[0].get_component_type_name())
        self.assertEquals("5.17", components[0].component_value["value"])
        self.assertEquals("float", components[0].component_value["type"])

    def testLoadsSimpleCanvas(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.type_repository.create(component_type_spec)

        def test_assign(dic):
            dic["value"] = "test_value"
            dic["value_type"] = "test_value"

        component_type.assign_default_value = test_assign

        text = """<canvas name="main"><component name="constant_1" type="Constant"><value>5.17</value><type>float</type></component></canvas>"""

        self.canvas_loader.load_component(text, 0)

        self.assertEquals(1, self.canvas_repository.count())
        self.assertEquals(1, self.component_repository.count())

        canvas_spec = CanvasSpecifications()
        canvas_spec.name = "main"
        canvases = self.canvas_repository.get(canvas_spec)

        self.assertEquals(1, len(canvases))
        self.assertEquals("main", canvases[0].name)

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"

        components = self.component_repository.get(spec)

        self.assertEquals(1, len(components))
        self.assertEquals("constant_1", components[0].name)
        self.assertEquals("Constant", components[0].get_component_type_name())
        self.assertEquals("5.17", components[0].component_value["value"])
        self.assertEquals("float", components[0].component_value["type"])

        self.assertEquals(1, canvases[0].count_components())
        self.assertEquals(components[0], canvases[0].components[0])
        self.assertEquals(canvases[0], components[0].canvas)

