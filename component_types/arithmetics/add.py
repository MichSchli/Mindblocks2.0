from model.component.component_type.component_type_model import ComponentTypeModel


class Add(ComponentTypeModel):

    name = "Add"

    def __init__(self):
        pass

    def out_degree(self):
        return 1

    def in_degree(self):
        return 2