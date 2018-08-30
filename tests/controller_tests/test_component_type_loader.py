import unittest

from Mindblocks.controller.component_type_loader.component_type_loader import ComponentTypeLoader
from Mindblocks.helpers.files.FilepathHandler import FilepathHandler
from Mindblocks.repository.component_type_repository.component_type_repository import ComponentTypeRepository
from Mindblocks.repository.identifier_repository.identifier_repository import IdentifierRepository
from tests.setup_holder import SetupHolder


class TestComponentTypeLoader(unittest.TestCase):

    def setUp(self):
        self.setup_holder = SetupHolder(load_default_types=False)

    def testLoadsSingleComponentType(self):
        filename = "arithmetics/add.py"
        filepath = self.setup_holder.filepath_handler.get_default_component_type_file(filename)
        self.setup_holder.component_type_loader.load_file(filepath)

        self.assertEqual(1, self.setup_holder.type_repository.count())

        c_type = list(self.setup_holder.type_repository.elements.values())[0]

        self.assertEqual("Add", c_type.name)

    def testLoadsSingleComponentTypeWithCorrectSockets(self):
        filename = "arithmetics/add.py"
        filepath = self.setup_holder.filepath_handler.get_default_component_type_file(filename)
        self.setup_holder.component_type_loader.load_file(filepath)

        self.assertEqual(1, self.setup_holder.type_repository.count())

        c_type = list(self.setup_holder.type_repository.elements.values())[0]

        self.assertEqual(["left", "right"], c_type.in_sockets)
        self.assertEqual(["output"], c_type.out_sockets)

    def testLoadsDirectory(self):
        filepath = self.setup_holder.filepath_handler.get_default_component_type_folder()
        self.setup_holder.component_type_loader.load_folder(filepath)

        self.assertLess(0, self.setup_holder.type_repository.count())

    def testLoadsDefaultDirectory(self):
        self.setup_holder.component_type_loader.load_default_folder()

        self.assertLess(0, self.setup_holder.type_repository.count())