class SequenceBatchValueModel:

    type = None
    item_shape = None

    batch_size = None
    max_length = None

    sequences = None
    sequence_lengths = None

    def __init__(self, type, item_shape):
        self.type = type
        self.item_shape = item_shape

        self.sequences = []
        self.sequence_lengths = []

    def get_sequences(self):
        return self.sequences

    def get_sequence_lengths(self):
        return self.sequence_lengths

    def assign(self, sequence_batch):
        self.sequences = sequence_batch
        self.sequence_lengths = [len(s) for s in sequence_batch]