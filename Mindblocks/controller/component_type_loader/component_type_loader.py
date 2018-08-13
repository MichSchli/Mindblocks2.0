import importlib
import importlib.machinery
import os


class ComponentTypeLoader:

    component_type_repository = None

    def __init__(self, filepath_handler, component_type_repository):
        self.component_type_repository = component_type_repository
        self.filepath_handler = filepath_handler

    def load_default_folder(self):
        filepath = self.filepath_handler.get_default_component_type_folder()
        self.load_folder(filepath)

    def load_folder(self, folder_location):
        all_subitems = os.listdir(folder_location)

        filtered_subitems = [item for item in all_subitems if self.filter_name(item)]
        absolute_subitems = [os.path.join(folder_location, d) for d in filtered_subitems]

        for f in absolute_subitems:
            if not os.path.isdir(f):
                self.load_file(f)
            else:
                self.load_folder(f)

    def load_file(self, f):
        f_string = ""
        class_file = open(f, 'r')
        for line in class_file:
            f_string += line
        class_file.close()

        if "(ComponentTypeModel)" not in f_string:
            return

        class_name_end_index = f_string.index("(ComponentTypeModel)")
        class_name = f_string[:class_name_end_index].split(" ")[-1]
        loaded_file = importlib.machinery.SourceFileLoader("module", f).load_module()
        component_type = getattr(loaded_file, class_name)()

        self.component_type_repository.add(component_type)

    def filter_name(self, name):
        if name.startswith('.') or name.startswith('_'):
            return False

        return True
