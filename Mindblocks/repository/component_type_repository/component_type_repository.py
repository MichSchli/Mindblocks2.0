from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository


class ComponentTypeRepository(AbstractRepository):

    def __initialize_model__(self):
        return ComponentTypeModel()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        return model

    def create_from_class(self, component_type_class, module_name=""):
        model = component_type_class()
        model.module_name = module_name
        self.__fill__(model)
        self.add(model)

    def get_module_dictionary(self):
        modules = {}

        for component_type in self.get_all():
            module_name = component_type.module_name
            if module_name not in modules:
                modules[module_name] = [component_type]
            else:
                modules[module_name].append(component_type)

        return modules