from Mindblocks.model.value_type.abstract_value_type import AbstractValueType


class SequenceBatchType(AbstractValueType):

    def __init__(self, type, dim, max_length):
        self.type = type
        self.max_length = max_length
        self.dim = dim