from Mindblocks.model.abstract.abstract_execution_model import AbstractExecutionModel


class ExecutionEdge(AbstractExecutionModel):

    source = None
    target = None

    def set_source(self, execution_out_socket):
        self.source = execution_out_socket

    def set_target(self, execution_in_socket):
        self.target = execution_in_socket

    def has_batches(self, mode):
        return self.source.has_batches(mode)

    def pull_type_model(self, mode):
        source_type = self.source.pull_type_model(mode)

        if self.value_model is not None and "cast" in self.value_model:
            return source_type.cast(self.value_model["cast"])
        else:
            return source_type

        return source_type

    def clear_caches(self):
        self.source.clear_caches()

    def pull(self, mode):
        source_value = self.source.pull(mode)

        if self.value_model is not None and "cast" in self.value_model:
            source_value = source_value.cast(self.value_model["cast"])

        if self.value_model is not None and "dropout_rate" in self.value_model:
            dropout_dim = None if "dropout_dim" not in self.value_model else self.value_model["dropout_dim"]
            source_value = source_value.apply_dropouts(self.value_model["dropout_rate"], dropout_dim=dropout_dim)

        return source_value

    def initialize(self, mode, tf_session_model):
        return self.source.initialize(mode, tf_session_model)

    # TODO: Make real value model (or create dummy component)
    def initialize_value(self, updated_dict, mode):
        value_model = {}

        if "cast" in updated_dict:
            value_model["cast"] = updated_dict["cast"][0][0]

        if "dropout_rate" in updated_dict:
            value_model["dropout_rate"] = float(updated_dict["dropout_rate"][0][0])
            if "dims" in updated_dict["dropout_rate"][0][1]:
                dims = int(updated_dict["dropout_rate"][0][1]["dims"])

                value_model["dropout_dim"] = dims

        return value_model

    def get_past(self):
        return self.source.get_past()

    def init_batches(self):
        self.source.init_batches()