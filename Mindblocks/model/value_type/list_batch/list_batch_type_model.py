from Mindblocks.model.tensorflow_placeholder.padded_tensor_placeholder_model import PaddedTensorPlaceholderModel
from Mindblocks.model.value_type.list_batch.list_batch_value_model import ListBatchValueModel


class ListBatchTypeModel:

    cached_seq_placeholder = None
    cached_len_placeholder = None

    def __init__(self, data_type, inner_shape, batch_size, max_length):
        self.data_type = data_type
        self.inner_shape = inner_shape
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
            self.inner_shape[-1] = int(dimension)

    def get_inner_dim(self):
        if len(self.inner_shape) == 0:
            return 1
        else:
            return self.inner_shape[-1]

    def extend_dims(self, dim):
        self.inner_shape.append(dim)

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
        return ListBatchValueModel()

    def format_from_tensorflow_output(self, output_tensors):
        value_model = self.initialize_value_model()
        value_model.assign(output_tensors[0])
        return value_model

    def is_value_type(self, test_type):
        return test_type == "list"