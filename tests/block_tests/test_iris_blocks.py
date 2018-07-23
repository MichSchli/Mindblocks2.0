import unittest

from controller.block_loader.block_loader import BlockLoader
from controller.block_loader.canvas_loader import CanvasLoader
from controller.block_loader.component_loader import ComponentLoader
from controller.block_loader.edge_loader import EdgeLoader
from controller.component_type_loader.component_type_loader import ComponentTypeLoader
from controller.graph_converter.graph_converter import GraphConverter
from helpers.files.FilepathHandler import FilepathHandler
from helpers.xml.xml_helper import XmlHelper
from repository.canvas_repository.canvas_repository import CanvasRepository
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.creation_component_repository.creation_component_repository import CreationComponentRepository
from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.graph.graph_repository import GraphRepository
from repository.identifier.identifier_repository import IdentifierRepository
from tests.setup_holder import SetupHolder
import numpy as np


class TestIrisBlocks(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder()

    def testReadsData(self):
        filename = "iris_tests/read_iris_data.xml"
        filepath = self.setup_holder.filepath_handler.get_test_block_path(filename)
        self.setup_holder.block_loader.load(filepath)

        component_spec = CreationComponentSpecifications()
        component_spec.name = "data_splitter"
        adder = self.setup_holder.component_repository.get(component_spec)[0]
        target_socket_1 = adder.get_out_socket("left")
        target_socket_2 = adder.get_out_socket("right")

        runs = [[target_socket_1, target_socket_2]]

        run_graphs = self.setup_holder.graph_converter.to_executable(runs)
        result = run_graphs[0].execute()

        self.assertEqual("0.2", result[0][0][-1])
        self.assertEqual("Iris-setosa", result[1][0][0])