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

    def initialize_value(self, value_dictionary):
        value = FileEmbeddingsValue(value_dictionary["file_path"][0][0])
        if "separator" in value_dictionary:
            value.separator = value_dictionary["separator"][0][0]
        return value

    def execute(self, input_dictionary, value, output_models, mode):
        value.load()

        output_models["index"].assign(value.get_index())
        output_models["vectors"].assign(value.get_vectors())

        return output_models

    def build_value_type_model(self, input_types, value):
        return {"index": IndexTypeModel(),
                "vectors": TensorTypeModel("float", [None, None])}


class FileEmbeddingsValue(ExecutionComponentValueModel):

    index = None
    vectors = None
    file_path = None
    separator = None

    def __init__(self, file_path):
        self.index = {"forward": {}, "backward": {}}
        self.file_path = file_path
        self.separator = ","

    def load(self):
        f = open(self.file_path, "r")
        self.vectors = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(self.separator)
                self.add_to_index(parts[0])
                self.add_to_vectors([float(t) for t in parts[1:]])

    def add_to_index(self, label):
        self.index["forward"][label] = len(self.index["forward"])
        self.index["backward"][len(self.index["backward"])] = label

    def add_to_vectors(self, vector):
        self.vectors.append(vector)

    def get_index(self):
        return self.index

    def get_vectors(self):
        return np.array(self.vectors, dtype=np.float32)
