import unittest

from controller.component_type_loader.component_type_loader import ComponentTypeLoader
from helpers.files.FilepathHandler import FilepathHandler
from repository.component_type_repository.component_type_repository import ComponentTypeRepository
from repository.identifier.identifier_repository import IdentifierRepository


class TestComponentTypeLoader(unittest.TestCase):

    def setUp(self):
        self.identifier_repository = IdentifierRepository()
        self.type_repository = ComponentTypeRepository(self.identifier_repository)
        self.filepath_handler = FilepathHandler()
        self.component_type_loader = ComponentTypeLoader(self.filepath_handler, self.type_repository)

    def testLoadsSingleComponentType(self):
        filename = "arithmetics/add.py"
        filepath = self.filepath_handler.get_default_component_type_file(filename)
        self.component_type_loader.load_file(filepath)

        self.assertEqual(1, self.type_repository.count())

        c_type = list(self.type_repository.elements.values())[0]

        self.assertEqual("Add", c_type.name)

    def testLoadsSingleComponentTypeWithCorrectSockets(self):
        filename = "arithmetics/add.py"
        filepath = self.filepath_handler.get_default_component_type_file(filename)
        self.component_type_loader.load_file(filepath)

        self.assertEqual(1, self.type_repository.count())

        c_type = list(self.type_repository.elements.values())[0]

        self.assertEqual(["left", "right"], c_type.in_sockets)
        self.assertEqual(["output"], c_type.out_sockets)

    def testLoadsDirectory(self):
        filepath = self.filepath_handler.get_default_component_type_folder()
        self.component_type_loader.load_folder(filepath)

        self.assertLess(0, self.type_repository.count())

    def testLoadsDefaultDirectory(self):
        self.component_type_loader.load_default_folder()

        self.assertLess(0, self.type_repository.count())