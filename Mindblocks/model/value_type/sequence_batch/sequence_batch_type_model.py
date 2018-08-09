from Mindblocks.model.value_type.sequence_batch.sequence_batch_value_model import SequenceBatchValueModel
from Mindblocks.model.value_type.tensor import tensor_type_model


class SequenceBatchTypeModel:

    type = None
    item_shape = None
    batch_size = None

    cached_seq_placeholder = None
    cached_len_placeholder = None

    def __init__(self, type, item_shape, batch_size):
        self.type = type
        self.item_shape = item_shape[:]
        self.batch_size = batch_size

    def initialize_value_model(self):
        return SequenceBatchValueModel(self.type, self.item_shape)

    def copy(self):
        return SequenceBatchTypeModel(self.type, self.item_shape, self.batch_size)

    def subsample(self, dimension):
        self.batch_size = dimension

    def set_inner_dim(self, dimension):
        if dimension == 1:
            self.item_shape = self.item_shape[:-1]
        else:
            self.item_shape[-1] = dimension

    def set_data_type(self, new_type):
        self.type = new_type

    def get_batch_size(self):
        return self.batch_size

    def extend_dims(self, dim):
        self.item_shape.append(dim)

    def get_tensorflow_placeholder(self):
        placeholder_model = self.initialize_value_model()
        placeholder_model.batch_size = 2
        placeholder_model.max_length = 5 # TODO: Hardcoded

        if self.cached_seq_placeholder is None:
            placeholder_model.initialize_as_tensorflow_placeholder()
            self.cached_seq_placeholder = placeholder_model.sequences
            self.cached_len_placeholder = placeholder_model.sequence_lengths
        else:
            placeholder_model.assign_with_lengths(self.cached_seq_placeholder, self.cached_len_placeholder)

        return placeholder_model

    def get_cached_placeholders(self):
        return [self.cached_seq_placeholder, self.cached_len_placeholder]

    def format_from_tensorflow_output(self, output_tensors):
        seqs = output_tensors[0]
        lens = output_tensors[1]

        fixed_seqs = []
        for i, length in enumerate(lens):
            fixed_seqs.append(seqs[i][:length])

        value_model = self.initialize_value_model()
        value_model.assign(fixed_seqs)
        return value_model

    def get_dimensions(self):
        return self.item_shape

    def get_single_token_type(self):
        return tensor_type_model.TensorTypeModel(self.type, self.item_shape)