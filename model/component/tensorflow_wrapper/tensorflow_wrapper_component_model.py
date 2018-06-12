from model.component.component_model import ComponentModel
from model.component.tensorflow_wrapper.tensorflow_wrapper_component_type_model import \
    TensorflowWrapperComponentTypeModel
from model.component.tensorflow_wrapper.tensorflow_wrapper_component_value_model import \
    TensorflowWrapperComponentValueModel


class TensorflowWrapperComponentModel(ComponentModel):

    def __init__(self, inner_component):
        self.value = TensorflowWrapperComponentValueModel(inner_component)
        self.component_type = TensorflowWrapperComponentTypeModel()

    def get_name(self):
        return "[TF]: " + self.value.inner_component.get_name()