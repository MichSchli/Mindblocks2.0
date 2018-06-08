import importlib
import importlib.machinery
import os

from model.component.component_type.component_type_model import ComponentTypeModel


class ComponentTypeLoader:


    def __init__(self):
        pass

    def load_component_type_folder(self, folder_location):
        all_subitems = os.listdir(folder_location)

        filtered_subitems = [item for item in all_subitems if self.filter_name(item)]
        absolute_subitems = [os.path.join(folder_location, d) for d in filtered_subitems]

        component_types = []
        for f in absolute_subitems:
            if not os.path.isdir(f):
                f_string = ""
                class_file = open(f, 'r')
                for line in class_file:
                    f_string += line
                class_file.close()
                class_name_end_index = f_string.index("(ComponentTypeModel)")
                class_name = f_string[:class_name_end_index].split(" ")[-1]

                loaded_file = importlib.machinery.SourceFileLoader("module", f).load_module()
                component_type = getattr(loaded_file, class_name)()
                component_types.append(component_type)
            else:
                component_types.extend(self.load_component_type_folder(f))

        return component_types

    def filter_name(self, name):
        if name.startswith('.') or name.startswith('_'):
            return False

        return True