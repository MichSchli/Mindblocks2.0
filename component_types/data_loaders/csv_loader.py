from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel


class CsvLoader(ComponentTypeModel):

    name = "CsvLoader"
    in_socket_names = []
    out_socket_names = ["output"]

    def __init__(self):
        pass

    def get_new_value(self):
        return CsvLoaderValue()

    def execute(self, in_sockets, value):
        return [value.read_array()]


class CsvLoaderValue(ComponentValueModel):

    array = None
    path = None
    separator = None

    def __init__(self):
        self.path = ""
        self.separator = "\t"

    def read_array(self):
        if self.array is None:
            self.array = []

            csv_file = open(self.path, 'r')
            for line in csv_file:
                line = line.strip()
                if line:
                    line = line.split(self.separator)
                    self.array.append(line)

        return self.array

    def load(self, value_lines):
        if "path" in value_lines:
            self.path = value_lines["path"][0][0]
        if "separator" in value_lines:
            self.separator = value_lines["separator"][0][0]

    def copy(self):
        copy = CsvLoaderValue()
        copy.path = self.path
        copy.separator = self.separator
        return copy

    def describe(self):
        return "path=\""+self.path+"\""
