class ExecutionOutSocket:

    cached_value = None
    cached_init_value = None
    cached_type = None

    should_use_placeholder_for_tensorflow_cache = None

    execution_component = None
    targets = None

    def __init__(self):
        self.targets = []

    def pull(self, mode):
        if self.cached_value is None:
            self.execution_component.execute(mode)

        return self.cached_value

    def set_determine_placeholders(self, p):
        self.should_use_placeholder_for_tensorflow_cache = p

    def set_cached_value(self, value):
        self.cached_value = value

    def set_cached_init_value(self, value):
        self.cached_init_value = value

    def add_target(self, in_socket):
        self.targets.append(in_socket)

    def clear_caches(self):
        self.cached_value = None
        self.execution_component.clear_caches()

    def has_batches(self):
        return self.execution_component.has_batches()

    def pull_type_model(self):
        if self.cached_type is None:
            self.execution_component.infer_type_models()
        return self.cached_type

    def set_cached_type(self, type_model):
        self.cached_type = type_model

    def describe_graph(self, indent=0):
        if self.execution_component is not None:
            self.execution_component.describe_graph(indent=indent+1)

    def init_batches(self):
        self.execution_component.init_batches()

    def should_use_placeholder_for_tensorflow(self):
        if self.should_use_placeholder_for_tensorflow_cache is None:
            self.execution_component.determine_placeholders()

        return self.should_use_placeholder_for_tensorflow_cache

    def initialize(self, mode):
        if self.cached_init_value is None:
            self.execution_component.initialize(mode)

        return self.cached_init_value