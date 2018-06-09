import numpy as np

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