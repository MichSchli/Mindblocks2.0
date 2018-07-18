from model.component_type.component_type_model import ComponentTypeModel


class Constant(ComponentTypeModel):

    name = "Constant"
    out_sockets = ["output"]
    languages = ["tensorflow", "python"]