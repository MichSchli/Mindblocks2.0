import numpy as np

from model.graph.value_type_model import ValueTypeModel


class Edge:

    satisfied = None
    source = None
    target = None
    source_socket = None
    target_socket = None
    cast_to = None

    def __init__(self, source, target):
        self.satisfied = False
        self.source = source
        self.target = target

    def copy(self):
        copy = Edge(self.source, self.target)
        copy.source_socket = self.source_socket
        copy.target_socket = self.target_socket
        copy.cast_to = self.cast_to

        return copy

    def get_value_type(self):
        source_value_type = self.source.get_out_value_types()[self.source_socket]
        return ValueTypeModel(source_value_type.type if self.cast_to is None else self.cast_to, source_value_type.dim)

    def __str__(self):
        return "( " + self.source.get_name() + " [" + str(self.source_socket) + "] -> " + self.target.get_name() + " [" + str(self.target_socket) + "] )"

    def is_satisfied(self):
        return self.satisfied

    def mark_satisfied(self, value):
        self.satisfied = value

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def put_value(self, value):
        self.value = value

    def get_value(self):
        if self.cast_to == "array:float":
            return np.array(self.value, dtype=np.float32)

        return self.value