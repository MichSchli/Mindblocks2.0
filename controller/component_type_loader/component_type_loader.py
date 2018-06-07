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
            f_string = ""
            for line in open(f, 'r'):
                f_string += line
            exec(f_string)
            component_types.append(ComponentTypeModel.__subclasses__()[-1]())

        subfolders = [d[0] for d in absolute_subitems if os.path.isdir(d[1])]
        for subfolder in subfolders:
            component_types.extend(self.load_component_type_folder(subfolder))

        return component_types

    def filter_name(self, name):
        if name.startswith('.') or name.startswith('_'):
            return False

        return True