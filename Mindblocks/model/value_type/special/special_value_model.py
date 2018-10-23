class SpecialValueModel:

    item = None

    def assign(self, item):
        self.item = item

    def get_value(self):
        return self.item

    def get_tensorflow_output_tensors(self):
        return [self.item]

    def format_for_program_output(self):
        return self.item