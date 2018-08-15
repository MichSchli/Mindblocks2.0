from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np

from Mindblocks.model.value_type.index.index_type_model import IndexTypeModel
from Mindblocks.model.value_type.old.index_type import IndexType
from Mindblocks.model.value_type.old.tensor_type import TensorType
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel


class FileEmbeddings(ComponentTypeModel):
    name = "FileEmbeddings"
    out_sockets = ["index", "vectors"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = FileEmbeddingsValue(value_dictionary["file_path"][0][0], int(value_dictionary["width"][0][0]))
        if "separator" in value_dictionary:
            value.separator = value_dictionary["separator"][0][0]

        if "token_list" in value_dictionary:
            pass

        if "stop_token" in value_dictionary:
            index = int(value_dictionary["stop_token"][0][1]["index"]) if "index" in value_dictionary["stop_token"][0][1] else 0
            value.add_stop_token(value_dictionary["stop_token"][0][0], index)

        return value

    def execute(self, input_dictionary, value, output_models, mode):
        if not value.loaded:
            value.load()

        output_models["index"].assign(value.get_index())
        output_models["vectors"].assign(value.get_vectors())

        return output_models

    def build_value_type_model(self, input_types, value):
        return {"index": IndexTypeModel(),
                "vectors": TensorTypeModel("float", [None, value.get_width()])}


class FileEmbeddingsValue(ExecutionComponentValueModel):

    index = None
    vectors = None
    file_path = None
    separator = None
    loaded = None

    stop_token = None
    stop_token_index = None

    def __init__(self, file_path, width):
        self.index = {"forward": {}, "backward": {}}
        self.file_path = file_path
        self.separator = ","
        self.width = width

        self.loaded = False

    def add_stop_token(self, token, index):
        self.stop_token = token
        self.stop_token_index = index

        if index == 0:
            self.add_to_index(token)

    def load(self):
        f = open(self.file_path, "r")
        self.vectors = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(self.separator)
                self.add_to_index(parts[0])
                self.add_to_vectors([float(t) for t in parts[1:]])

        self.loaded = True

        if self.stop_token is not None:
            self.insert_stop_token()

    def add_to_index(self, label):
        self.index["forward"][label] = len(self.index["forward"])
        self.index["backward"][len(self.index["backward"])] = label

        if self.stop_token is not None and len(self.index["forward"]) == self.stop_token_index:
            self.index["forward"][self.stop_token] = len(self.index["forward"])
            self.index["backward"][len(self.index["backward"])] = self.stop_token

    def add_to_vectors(self, vector):
        self.vectors.append(vector)

    def insert_stop_token(self):
        stop_token_vector = [0] * self.width
        self.vectors.insert(self.stop_token_index, stop_token_vector)

    def get_index(self):
        return self.index

    def get_vectors(self):
        return np.array(self.vectors, dtype=np.float32)

    def get_width(self):
        return self.width