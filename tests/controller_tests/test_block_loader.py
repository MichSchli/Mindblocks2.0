import unittest

from controller.block_loader.canvas_loader import CanvasLoader
from controller.block_loader.component_loader import ComponentLoader
from controller.block_loader.edge_loader import EdgeLoader
from helpers.xml.xml_helper import XmlHelper
from repository.canvas_repository.canvas_repository import CanvasRepository
from repository.canvas_repository.canvas_specifications import CanvasSpecifications
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.identifier.identifier_repository import IdentifierRepository


class TestBlockLoader(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.canvas_repository = CanvasRepository(self.identifier_repository)
        self.graph_repository = GraphRepository(self.identifier_repository)
        self.component_repository = CreationComponentRepository(self.identifier_repository,
                                                                self.type_repository,
                                                                self.canvas_repository,
                                                                self.graph_repository)

        self.xml_helper = XmlHelper()
        self.component_loader = ComponentLoader(self.xml_helper, self.component_repository)
        self.edge_loader = EdgeLoader(self.xml_helper, self.graph_repository, self.component_repository)
        self.canvas_loader = CanvasLoader(self.xml_helper, self.component_loader, self.edge_loader, self.canvas_repository)

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

        self.canvas_loader.load_canvas(text, 0)

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

    def testLoadsSimpleCanvasWithTwoComponents(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.type_repository.create(component_type_spec)

        def test_assign(dic):
            dic["value"] = "test_value"
            dic["value_type"] = "test_value"

        component_type.assign_default_value = test_assign

        text = """<canvas name="main">
        <component name="constant_1" type="Constant"><value>5.17</value><type>float</type></component>
        <component name="constant_2" type="Constant"><value>8.14</value><type>float</type></component>
        </canvas>"""

        self.canvas_loader.load_canvas(text, 0)

        self.assertEquals(1, self.canvas_repository.count())
        self.assertEquals(2, self.component_repository.count())

        canvas_spec = CanvasSpecifications()
        canvas_spec.name = "main"
        canvases = self.canvas_repository.get(canvas_spec)

        self.assertEquals(1, len(canvases))
        self.assertEquals("main", canvases[0].name)
        self.assertEquals(2, canvases[0].count_components())

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"

        components = self.component_repository.get(spec)

        self.assertEquals(1, len(components))
        self.assertEquals("constant_1", components[0].name)
        self.assertEquals("Constant", components[0].get_component_type_name())
        self.assertEquals("5.17", components[0].component_value["value"])
        self.assertEquals("float", components[0].component_value["type"])
        self.assertEquals(canvases[0], components[0].canvas)

        spec.name = "constant_2"

        components = self.component_repository.get(spec)

        self.assertEquals(1, len(components))
        self.assertEquals("constant_2", components[0].name)
        self.assertEquals("Constant", components[0].get_component_type_name())
        self.assertEquals("8.14", components[0].component_value["value"])
        self.assertEquals("float", components[0].component_value["type"])
        self.assertEquals(canvases[0], components[0].canvas)

    def testLoadsSimpleCanvasWithEdge(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.type_repository.create(component_type_spec)
        component_type.out_sockets = ["out"]

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Printer"
        component_type = self.type_repository.create(component_type_spec)
        component_type.in_sockets = ["in"]

        text = """<canvas name="main">
        <component name="constant_1" type="Constant"></component>
        <component name="printer" type="Printer"></component>
        <edge><source socket=out>constant_1</source><target socket=in>printer</target></edge>
        </canvas>"""

        self.canvas_loader.load_canvas(text, 0)

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"
        components = self.component_repository.get(spec)
        graph = components[0].graph

        self.assertEquals(1, len(graph.get_edges()))
        self.assertEquals(2, len(graph.get_vertices()))

        edge = graph.get_edges()[0]
        self.assertEquals("constant_1", edge.get_source_component_name())
        self.assertEquals("printer", edge.get_target_component_name())

