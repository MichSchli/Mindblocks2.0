class Edge:

    satisfied = None
    source = None
    target = None

    def __init__(self, source, target):
        self.satisfied = False
        self.source = source
        self.target = target

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
        return self.value