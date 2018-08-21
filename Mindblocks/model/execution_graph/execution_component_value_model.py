class ExecutionComponentValueModel:

    component_name = None
    component_mode = None

    def get_populate_items(self):
        return []

    def init_batches(self):
        self.has_batch = True

    def set_component_name(self, name, mode):
        self.component_name = name
        self.component_mode = mode

    def get_name(self):
        return str(self.component_name) + "-" + str(self.component_mode)

    def initialize_tensorflow_variables(self, tensorflow_session_model):
        pass
