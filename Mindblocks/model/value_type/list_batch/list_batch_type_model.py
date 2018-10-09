from Mindblocks.model.tensorflow_placeholder.padded_tensor_placeholder_model import PaddedTensorPlaceholderModel
from Mindblocks.model.value_type.list_batch.list_batch_value_model import ListBatchValueModel


class ListBatchTypeModel:

    cached_seq_placeholder = None
    cached_len_placeholder = None

    def __init__(self, data_type, inner_shape, batch_size, max_length):
        max_length = None
        self.data_type = data_type
        self.inner_shape = inner_shape[:]
        self.batch_size = batch_size
        self.max_length = max_length
        self.cached_casts = {}
        self.placeholder_manager = PaddedTensorPlaceholderModel(
            batch_size,
            max_length,
            inner_shape,
            data_type
        )

    def get_dimensions(self):
        return self.inner_shape

    def copy(self):
        return ListBatchTypeModel(
            self.data_type,
            self.inner_shape,
            self.batch_size,
            self.max_length
        )

    def get_cached_placeholders(self):
        return self.placeholder_manager.get_placeholders()

    def get_tensorflow_placeholder(self):
        placeholder_model = self.initialize_value_model()
        placeholder_model.max_length = None

        data, lengths = self.placeholder_manager.get_placeholders()
        placeholder_model.item = data
        placeholder_model.lengths = lengths

        return placeholder_model

    def set_data_type(self, data_type):
        self.placeholder_manager.type = data_type
        self.data_type = data_type

    def subsample(self, dim):
        self.batch_size = dim

    def set_inner_dim(self, dimension):
        if dimension == 1:
            self.inner_shape = self.inner_shape[:-1]
        else:
            self.inner_shape[-1] = dimension

    def get_inner_dim(self):
        if len(self.inner_shape) == 0:
            return 1
        else:
            return self.inner_shape[-1]

    def extend_dims(self, dim):
        self.inner_shape.append(dim)

    def set_dim(self, i, v):
        if i > 1:
            self.inner_shape[i-2] = v

    def remove_dim(self, i):
        if i > 1:
            del self.inner_shape[i-2]

    cached_casts = None

    def cast(self, new_type):
        if new_type not in self.cached_casts:
            self.cached_casts[new_type] = ListBatchTypeModel(
                new_type,
                self.inner_shape,
                self.batch_size,
                self.max_length
            )
        return self.cached_casts[new_type]

    def initialize_value_model(self):
        value_model = ListBatchValueModel(self.data_type, self.inner_shape)
        value_model.maximum_length = self.max_length
        return value_model

    def set_batch_size(self, size):
        self.batch_size = size

    def format_from_tensorflow_output(self, output_tensors):
        seqs = output_tensors[0]
        lens = output_tensors[1]

        fixed_seqs = []
        for i, length in enumerate(lens):
            fixed_seqs.append(seqs[i][:length])

        value_model = self.initialize_value_model()
        value_model.assign(fixed_seqs)
        return value_model

    def is_value_type(self, test_type):
        return test_type == "list"