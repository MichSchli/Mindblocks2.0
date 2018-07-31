import unittest

from controller.block_loader.block_loader import BlockLoader
from controller.block_loader.canvas_loader import CanvasLoader
from controller.block_loader.component_loader import ComponentLoader
from controller.block_loader.edge_loader import EdgeLoader
from helpers.files.FilepathHandler import FilepathHandler
from helpers.xml.xml_helper import XmlHelper
from repository.canvas_repository.canvas_repository import CanvasRepository
from repository.canvas_repository.canvas_specifications import CanvasSpecifications
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.component_type_repository.component_type_specifications import ComponentTypeSpecifications
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.identifier.identifier_repository import IdentifierRepository
from repository.variable_repository.variable_specifications import VariableSpecifications
from tests.setup_holder import SetupHolder


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
        self.block_loader = BlockLoader(self.xml_helper, self.canvas_loader, None)

        self.filepath_handler = FilepathHandler()

        self.setup_holder = SetupHolder(load_default_types=False)

    def testLoadsVariable(self):
        text = """<variable name="test_variable"><default_value>45</default_value></variable>"""
        self.setup_holder.variable_loader.load_variable(text, 0)

        spec = VariableSpecifications()
        spec.name = "test_variable"

        variables = self.setup_holder.variable_repository.get(spec)

        self.assertEqual(1, len(variables))
        self.assertEqual("test_variable", variables[0].name)
        self.assertEqual("45", variables[0].get_value())

    def testLoadsVariableWithDefaultAndTrainValues(self):
        text = """<variable name="test_variable"><default_value>45</default_value><train_value>86</train_value></variable>"""
        self.setup_holder.variable_loader.load_variable(text, 0)

        spec = VariableSpecifications()
        spec.name = "test_variable"

        variables = self.setup_holder.variable_repository.get(spec)

        self.assertEqual(1, len(variables))
        self.assertEqual("test_variable", variables[0].name)
        self.assertEqual("45", variables[0].get_value())
        self.assertEqual("45", variables[0].get_value(mode="test"))
        self.assertEqual("86", variables[0].get_value(mode="train"))

    def testLoadsVariableWithValidValue(self):
        text = """<variable name="test_variable"><validate_value>86</validate_value></variable>"""
        self.setup_holder.variable_loader.load_variable(text, 0)

        spec = VariableSpecifications()
        spec.name = "test_variable"

        variables = self.setup_holder.variable_repository.get(spec)

        self.assertEqual(1, len(variables))
        self.assertEqual("test_variable", variables[0].name)
        self.assertIsNone(variables[0].get_value())
        self.assertIsNone(variables[0].get_value(mode="test"))
        self.assertEqual("86", variables[0].get_value(mode="validate"))

    def testConfigurationLoaderLoadsVariables(self):
        text = """<configuration>
        <variable name="test_variable"><default_value>45</default_value></variable>
        <variable name="test_variable_2"><default_value>16.2</default_value></variable>
        </configuration>"""

        self.setup_holder.configuration_loader.load_configuration(text, 0)

        spec = VariableSpecifications()

        variables = self.setup_holder.variable_repository.get(spec)

        self.assertEqual(2, len(variables))
        variable_names = [v.name for v in variables]

        self.assertIn("test_variable", variable_names)
        self.assertIn("test_variable_2", variable_names)

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

        self.assertEqual(1, self.component_repository.count())

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"

        components = self.component_repository.get(spec)

        self.assertEqual(1, len(components))
        self.assertEqual("constant_1", components[0].name)
        self.assertEqual("Constant", components[0].get_component_type_name())
        self.assertEqual("5.17", components[0].component_value["value"])
        self.assertEqual("float", components[0].component_value["type"])

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

        self.assertEqual(1, self.canvas_repository.count())
        self.assertEqual(1, self.component_repository.count())

        canvas_spec = CanvasSpecifications()
        canvas_spec.name = "main"
        canvases = self.canvas_repository.get(canvas_spec)

        self.assertEqual(1, len(canvases))
        self.assertEqual("main", canvases[0].name)

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"

        components = self.component_repository.get(spec)

        self.assertEqual(1, len(components))
        self.assertEqual("constant_1", components[0].name)
        self.assertEqual("Constant", components[0].get_component_type_name())
        self.assertEqual("5.17", components[0].component_value["value"])
        self.assertEqual("float", components[0].component_value["type"])

        self.assertEqual(1, canvases[0].count_components())
        self.assertEqual(components[0], canvases[0].components[0])
        self.assertEqual(canvases[0], components[0].canvas)

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

        self.assertEqual(1, self.canvas_repository.count())
        self.assertEqual(2, self.component_repository.count())

        canvas_spec = CanvasSpecifications()
        canvas_spec.name = "main"
        canvases = self.canvas_repository.get(canvas_spec)

        self.assertEqual(1, len(canvases))
        self.assertEqual("main", canvases[0].name)
        self.assertEqual(2, canvases[0].count_components())

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"

        components = self.component_repository.get(spec)

        self.assertEqual(1, len(components))
        self.assertEqual("constant_1", components[0].name)
        self.assertEqual("Constant", components[0].get_component_type_name())
        self.assertEqual("5.17", components[0].component_value["value"])
        self.assertEqual("float", components[0].component_value["type"])
        self.assertEqual(canvases[0], components[0].canvas)

        spec.name = "constant_2"

        components = self.component_repository.get(spec)

        self.assertEqual(1, len(components))
        self.assertEqual("constant_2", components[0].name)
        self.assertEqual("Constant", components[0].get_component_type_name())
        self.assertEqual("8.14", components[0].component_value["value"])
        self.assertEqual("float", components[0].component_value["type"])
        self.assertEqual(canvases[0], components[0].canvas)

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

        self.assertEqual(1, len(graph.get_edges()))
        self.assertEqual(2, len(graph.get_vertices()))

        edge = graph.get_edges()[0]
        self.assertEqual("constant_1", edge.get_source_component_name())
        self.assertEqual("printer", edge.get_target_component_name())

    def testLoadsFullBlock(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.setup_holder.type_repository.create(component_type_spec)
        component_type.out_sockets = ["out"]

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Printer"
        component_type = self.setup_holder.type_repository.create(component_type_spec)
        component_type.in_sockets = ["in"]

        text = """<block>
        <canvas name="main">
        <component name="constant_1" type="Constant"></component>
        <component name="printer" type="Printer"></component>
        <edge><source socket=out>constant_1</source><target socket=in>printer</target></edge>
        </canvas>
        </block>"""

        self.setup_holder.block_loader.load_block(text, 0)

        self.assertEqual(1, self.setup_holder.canvas_repository.count())
        self.assertEqual(2, self.setup_holder.component_repository.count())

    def testLoadsFullWithVariable(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.setup_holder.type_repository.create(component_type_spec)
        component_type.out_sockets = ["out"]

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Printer"
        component_type = self.setup_holder.type_repository.create(component_type_spec)
        component_type.in_sockets = ["in"]

        text = """<block>
        <configuration>
        <variable name="test_variable"><default_value>45</default_value></variable>
        </configuration>
        <canvas name="main">
        <component name="constant_1" type="Constant"></component>
        <component name="printer" type="Printer"></component>
        <edge><source socket=out>constant_1</source><target socket=in>printer</target></edge>
        </canvas>
        </block>"""

        self.setup_holder.block_loader.load_block(text, 0)

        self.assertEqual(1, self.setup_holder.canvas_repository.count())
        self.assertEqual(1, self.setup_holder.variable_repository.count())

    def testLoadsFullBlockTwoCanvases(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.type_repository.create(component_type_spec)
        component_type.out_sockets = ["out"]

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Printer"
        component_type = self.type_repository.create(component_type_spec)
        component_type.in_sockets = ["in"]

        text = """<block>
        <canvas name="main">
        <component name="constant_1" type="Constant"></component>
        <component name="printer" type="Printer"></component>
        <edge><source socket=out>constant_1</source><target socket=in>printer</target></edge>
        </canvas>
        <canvas name="secondary">
        <component name="constant_1_s" type="Constant"></component>
        <component name="printer_s" type="Printer"></component>
        <edge><source socket=out>constant_1_s</source><target socket=in>printer_s</target></edge>
        </canvas>
        </block>"""

        self.block_loader.load_block(text, 0)

        self.assertEqual(2, self.canvas_repository.count())

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"
        component = self.component_repository.get(spec)[0]
        self.assertEqual("main", component.canvas.name)

        spec = CreationComponentSpecifications()
        spec.name = "constant_1_s"
        component = self.component_repository.get(spec)[0]
        self.assertEqual("secondary", component.canvas.name)

    def testLoadsFromFile(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.type_repository.create(component_type_spec)
        component_type.out_sockets = ["output"]

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Add"
        component_type = self.type_repository.create(component_type_spec)
        component_type.in_sockets = ["left", "right"]

        filename = "add_constants.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)

        self.block_loader.load(filepath)

        self.assertEqual(1, self.canvas_repository.count())
        self.assertEqual(3, self.component_repository.count())
        self.assertEqual(2, list(self.graph_repository.elements.values())[0].count_edges())

    def testLoadsLanguagesCorrect(self):
        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Constant"
        component_type = self.type_repository.create(component_type_spec)
        component_type.out_sockets = ["output"]
        component_type.languages = ["test_language"]

        component_type_spec = ComponentTypeSpecifications()
        component_type_spec.name = "Add"
        component_type = self.type_repository.create(component_type_spec)
        component_type.in_sockets = ["left", "right"]
        component_type.languages = ["tensorflow", "python"]

        filename = "add_constants.xml"
        filepath = self.filepath_handler.get_test_block_path(filename)
        self.block_loader.load(filepath)

        spec = CreationComponentSpecifications()
        spec.name = "constant_1"
        component = self.component_repository.get(spec)[0]
        self.assertEqual("test_language", component.language)

        spec = CreationComponentSpecifications()
        spec.name = "adder"
        component = self.component_repository.get(spec)[0]
        self.assertEqual("python", component.language)