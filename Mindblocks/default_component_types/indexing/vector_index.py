from Mindblocks.helpers.logging.logger_factory import LoggerFactory
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np

from Mindblocks.model.value_type.index.index_type_model import IndexTypeModel
import tensorflow as tf

from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class VectorIndex(ComponentTypeModel):
    name = "VectorIndex"
    out_sockets = ["index", "vectors"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = VectorIndexValue()

        if "width" in value_dictionary:
            value.set_width(int(value_dictionary["width"][0][0]))

        if "length" in value_dictionary:
            value.set_number_of_vectors(int(value_dictionary["length"][0][0]))

        if "trainable" in value_dictionary:
            value.set_trainable(value_dictionary["trainable"][0][0])

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):

        output_models["index"].assign(value.get_index(), length_list=None)
        output_models["vectors"].assign(value.get_vectors(), length_list=None)

        return output_models

    def initialize(self, input_dictionary, value, output_value_models, tensorflow_session_model):
        if value.get_vectors() is None:
            value.initialize_vectors()
        output_value_models["vectors"].assign(value.get_vectors(), length_list=None)

        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        vector_model = SoftTensorTypeModel([None, value.get_width()], string_type="float")
        return {"index": IndexTypeModel(),
                "vectors": vector_model}

    def determine_placeholders(self, value, out_socket_names):
        default = {k: True for k in out_socket_names}

        default["vectors"] = False

        return default


class VectorIndexValue(ExecutionComponentValueModel):

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
    stop_token_index = None
    trainable = None

    def __init__(self):
        self.index = {"forward": {}, "backward": {}, "unk_token": None}
        self.separator = ","
        self.loaded = False
        self.next_item_pointer = 0
        self.special_symbols = []
        self.trainable = True

    def set_width(self, width):
        self.width = width

    """
    Initialization:
    """

    def initialize_vectors(self):
        self.vectors = np.random.uniform(-0.1, 0.1, size=(self.length, self.width)).astype(np.float32)

        if self.should_output_tensorflow():
            self.format_output_to_tensorflow()

    def set_number_of_vectors(self, number_of_vectors):
        self.length = number_of_vectors

    def dimensions_known(self):
        return self.length is not None

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

    def set_unk_token(self, token):
        self.index["unk_token"] = token

    def add_unk_token(self, token, index):
        self.set_unk_token(token)
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
    Get values:
    """

    def get_index(self):
        return self.index

    def get_vectors(self):
        return self.vectors

    def get_width(self):
        return self.width

    def get_length(self):
        return self.length

    """
    Utils:
    """

    def count_parameters(self):
        if self.trainable and self.trainable != "feed":
            return self.length * self.width
        else:
            return 0
