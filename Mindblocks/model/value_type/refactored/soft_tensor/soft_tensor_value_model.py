class SoftTensorValueModel:

    dimensions = None
    string_type = None

    max_lengths = None
    soft_length_tensors = None

    tensor = None
    length_list = None

    def __init__(self, dimensions, string_type, max_lengths):
        self.dimensions = dimensions
        self.string_type = string_type
        self.max_lengths = max_lengths
        self.soft_length_tensors = max_lengths

    def get_dimensions(self):
        return self.dimensions

    def assign(self, tensor, length_list=None):
        self.tensor = tensor

        if length_list is None:
            self.soft_length_tensors = self.get_max_lengths()
        else:
            self.soft_length_tensors = length_list

    def get_value(self):
        return self.tensor

    def get_lengths(self):
        return self.soft_length_tensors

    def get_max_lengths(self):
        return self.max_lengths