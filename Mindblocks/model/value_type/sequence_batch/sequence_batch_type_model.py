from Mindblocks.model.value_type.sequence_batch.sequence_batch_value_model import SequenceBatchValueModel


class SequenceBatchTypeModel:

    type = None
    item_shape = None

    def __init__(self, type, item_shape):
        self.type = type
        self.item_shape = item_shape

    def initialize_value_model(self):
        return SequenceBatchValueModel(self.type, self.item_shape)