from Mindblocks.model.value_type.abstract_value_type import AbstractValueType


class TensorType(AbstractValueType):

    def __init__(self, type, dims):
        self.type = type
        self.dims = dims