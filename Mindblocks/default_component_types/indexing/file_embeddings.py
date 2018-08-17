from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np

from Mindblocks.model.value_type.index.index_type_model import IndexTypeModel
import tensorflow as tf
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
            value.token_list = value_dictionary["token_list"][0][0]

        if "stop_token" in value_dictionary:
            index = int(value_dictionary["stop_token"][0][1]["index"]) if "index" in value_dictionary["stop_token"][0][1] else 0
            value.add_stop_token(value_dictionary["stop_token"][0][0], index)

        if "unk_token" in value_dictionary:
            index = int(value_dictionary["unk_token"][0][1]["index"]) if "index" in value_dictionary["unk_token"][0][1] else value.stop_token_index + 1
            value.add_unk_token(value_dictionary["unk_token"][0][0], index)

        return value

    def execute(self, input_dictionary, value, output_models, mode):

        output_models["index"].assign(value.get_index())
        output_models["vectors"].assign(value.get_vectors())

        return output_models

    def initialize(self, input_dictionary, value, output_value_models):
        if not value.loaded:
            value.load()
        output_value_models["vectors"].assign(value.get_vectors())

        return output_value_models

    def build_value_type_model(self, input_types, value):
        vector_model = TensorTypeModel("float", [None, value.get_width()])
        return {"index": IndexTypeModel(),
                "vectors": vector_model}

    def determine_placeholders(self, value, out_socket_names):
        default = {k: True for k in out_socket_names}

        default["vectors"] = False

        return default


class FileEmbeddingsValue(ExecutionComponentValueModel):

    index = None
    vectors = None
    file_path = None
    separator = None
    loaded = None

    stop_token = None
    stop_token_index = None

    unk_token = None
    unk_token_index = None

    token_list = None

    def __init__(self, file_path, width):
        self.index = {"forward": {}, "backward": {}, "unk_token": None}
        self.file_path = file_path
        self.separator = ","
        self.width = width

        self.loaded = False

    def add_stop_token(self, token, index):
        self.stop_token = token
        self.stop_token_index = index

        if index == 0:
            self.add_to_index(token)

    def add_unk_token(self, token, index):
        self.unk_token = token
        self.unk_token_index = index

        self.index["unk_token"] = token

        if index == len(self.index["forward"]):
            self.add_to_index(token)

    def load_token_list(self):
        token_list = []
        f = open(self.token_list)
        for line in f:
            line = line.strip()
            if line:
                token_list.append(line)

        return token_list

    def load(self):
        token_list = None
        if self.token_list is not None:
            token_list = self.load_token_list()

        f = open(self.file_path, "r")
        self.vectors = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(self.separator)

                if token_list is not None and parts[0] not in token_list:
                    continue

                self.add_to_index(parts[0])
                self.add_to_vectors([float(t) for t in parts[1:]])

                if token_list is not None and len(self.vectors) == len(token_list):
                    break

        self.loaded = True

        if self.stop_token is not None:
            self.insert_stop_token()

        if self.unk_token is not None:
            self.insert_unk_token()

        self.vectors = tf.Variable(np.array(self.vectors, dtype=np.float32), trainable=False)

    def add_to_index(self, label):
        self.index["forward"][label] = len(self.index["forward"])
        self.index["backward"][len(self.index["backward"])] = label

        if self.stop_token is not None and len(self.index["forward"]) == self.stop_token_index:
            self.index["forward"][self.stop_token] = len(self.index["forward"])
            self.index["backward"][len(self.index["backward"])] = self.stop_token

        if self.unk_token is not None and len(self.index["forward"]) == self.unk_token_index:
            self.index["forward"][self.unk_token] = len(self.index["forward"])
            self.index["backward"][len(self.index["backward"])] = self.unk_token

    def add_to_vectors(self, vector):
        self.vectors.append(vector)

    def insert_stop_token(self):
        stop_token_vector = [0] * self.width
        self.vectors.insert(self.stop_token_index, stop_token_vector)

    def insert_unk_token(self):
        unk_token_vector = [-1] * self.width
        self.vectors.insert(self.unk_token_index, unk_token_vector)

    def get_index(self):
        return self.index

    def get_vectors(self):
        return self.vectors

    def get_width(self):
        return self.width