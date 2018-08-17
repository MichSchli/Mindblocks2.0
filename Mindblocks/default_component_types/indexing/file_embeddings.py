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
            value.use_vocabulary(value_dictionary["token_list"][0][0])

        if "stop_token" in value_dictionary:
            index = int(value_dictionary["stop_token"][0][1]["index"]) if "index" in value_dictionary["stop_token"][0][1] else 0
            value.add_stop_token(value_dictionary["stop_token"][0][0], index)

        if "unk_token" in value_dictionary:
            index = int(value_dictionary["unk_token"][0][1]["index"]) if "index" in value_dictionary["unk_token"][0][1] else value.stop_token_index + 1
            value.add_unk_token(value_dictionary["unk_token"][0][0], index)

        if "trainable" in value_dictionary:
            value.set_trainable(value_dictionary["trainable"][0][0])

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

    length = None
    width = None
    vocabulary_file_path = None
    vocabulary = None
    special_symbols = None
    index = None
    vectors = None
    file_path = None
    separator = None
    loaded = None
    next_item_pointer = None
    stop_token = None
    trainable = None

    def __init__(self, file_path, width):
        self.index = {"forward": {}, "backward": {}, "unk_token": None}
        self.file_path = file_path
        self.separator = ","
        self.width = width
        self.loaded = False
        self.next_item_pointer = 0
        self.special_symbols = []
        self.trainable = False

    """
    Vocabulary:
    """

    def load_token_list(self, filepath):
        token_list = []
        with open(filepath, 'r') as token_file:
            for line in token_file:
                line = line.strip()
                if line:
                    token_list.append(line)

        return token_list

    def use_vocabulary(self, filepath):
        self.vocabulary_file_path = filepath

    def uses_vocabulary(self):
        return self.vocabulary_file_path is not None

    def load_vocabulary(self):
        self.vocabulary = self.load_token_list(self.vocabulary_file_path)
        self.length = len(self.vocabulary) + self.count_special_symbols()

    def free_vocabulary(self):
        self.vocabulary = None

    def in_vocabulary(self, item):
        return item in self.vocabulary

    """
    Initialization:
    """

    def count_lines(self):
        with open(self.file_path, 'r') as f:
            num_lines = sum(1 for _ in f)
        return num_lines

    def set_number_of_vectors(self, number_of_vectors):
        self.length = number_of_vectors

    def dimensions_known(self):
        return self.length is not None

    def initialize_vectors(self):
        self.vectors = np.zeros((self.length, self.width), dtype=np.float32)

    """
    Special characters:
    """

    def add_special_symbol(self, token, index, vector):
        self.special_symbols.append((token, index, vector))

    def count_special_symbols(self):
        return len(self.special_symbols)

    def populate_special_characters(self):
        for token, index, vector in self.special_symbols:
            self.add_to_index(token, index)
            self.add_vector(vector, index)

    def add_stop_token(self, token, index):
        self.stop_token = token
        self.add_special_symbol(token, index, np.zeros(self.width, dtype=np.float32))

    def add_unk_token(self, token, index):
        self.index["unk_token"] = token
        self.add_special_symbol(token, index, np.ones(self.width, dtype=np.float32)*-1)

    """
    Appending:
    """

    def add_to_index(self, token, index):
        self.index["forward"][token] = index
        self.index["backward"][index] = token

    def add(self, token):
        while self.next_item_pointer in self.index["backward"]:
            self.next_item_pointer += 1

        self.add_to_index(token, self.next_item_pointer)
        return self.next_item_pointer

    def add_vector(self, vector, index):
        self.vectors[index] = vector

    """
    Tensorflow formatting:
    """

    def format_output_to_tensorflow(self):
        self.vectors = tf.Variable(np.array(self.vectors, dtype=np.float32), trainable=self.trainable)

    def should_output_tensorflow(self):
        return self.trainable != "feed"

    def set_trainable(self, value):
        if value == "feed":
            self.trainable = value
        else:
            self.trainable = value == "True"

    """
    Loading:
    """

    def has_loaded_all_items(self):
        while self.next_item_pointer in self.index["backward"]:
            self.next_item_pointer += 1

        return self.next_item_pointer == self.length

    def load(self):
        if self.uses_vocabulary():
            self.load_vocabulary()

        if not self.dimensions_known():
            number_of_vectors = self.count_lines() + self.count_special_symbols()
            self.set_number_of_vectors(number_of_vectors)

        self.initialize_vectors()
        self.populate_special_characters()

        with open(self.file_path, "r") as vector_file:
            for line in vector_file:
                line = line.strip()
                if line:
                    parts = line.split(self.separator)

                    if self.uses_vocabulary() and not self.in_vocabulary(parts[0]):
                        continue

                    index = self.add(parts[0])
                    vector = np.zeros(self.width, dtype=np.float32)
                    for i in range(1, min(self.width, len(parts))):
                        vector[i] = float(parts[i])

                    self.add_vector(vector, index)

                    if self.has_loaded_all_items():
                        break

        self.loaded = True

        if self.should_output_tensorflow():
            self.format_output_to_tensorflow()

        if self.uses_vocabulary():
            self.free_vocabulary()

    """
    Get values:
    """

    def get_index(self):
        return self.index

    def get_vectors(self):
        return self.vectors

    def get_width(self):
        return self.width