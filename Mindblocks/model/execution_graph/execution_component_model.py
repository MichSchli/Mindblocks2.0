from Mindblocks.model.abstract.abstract_model import AbstractModel
from Mindblocks.model.abstract.abstract_execution_model import AbstractExecutionModel


class ExecutionComponentModel(AbstractModel, AbstractExecutionModel):

    out_sockets = None
    in_sockets = None
    identifier = None
    component_identifier = None
    language = None

    name = None
    mode = None

    def __init__(self):
        self.out_sockets = {}
        self.in_sockets = {}

    def get_language(self):
        return self.language

    def always_require_unique(self, mode):
        return self.has_referenced_graphs(mode)

    def initialize_value(self, corrected_value_dictionary, mode):
        value = self.execution_type.initialize_value(corrected_value_dictionary, mode)
        value.language = self.language
        value.set_component_name(self.name, self.mode)
        return value

    def add_out_socket(self, key, socket):
        self.out_sockets[key] = socket

    def add_in_socket(self, key, socket):
        self.in_sockets[key] = socket

    def execute(self, mode):
        input_dictionary = {k : in_socket.pull(mode) for k,in_socket in self.in_sockets.items()}
        output_value_models = {k : type_model.initialize_value_model() for k,type_model in self.output_type_models.items()}

        output_dictionary = self.execution_type.execute(self, input_dictionary, self.value_model, output_value_models, mode)

        for k,v in output_dictionary.items():
            self.out_sockets[k].set_cached_value(v)

    def initialize(self, mode, tensorflow_session_model):
        input_dictionary = {k: in_socket.initialize(mode, tensorflow_session_model) for k, in_socket in self.in_sockets.items()}
        output_value_models = {k: type_model.initialize_value_model() for k, type_model in
                               self.output_type_models.items()}

        output_dictionary = self.execution_type.initialize(input_dictionary,
                                                           self.value_model,
                                                           output_value_models,
                                                           tensorflow_session_model)

        for k,v in output_dictionary.items():
            self.out_sockets[k].set_cached_init_value(v)

    def determine_placeholders(self):
        d = self.execution_type.determine_placeholders(self.value_model, self.output_type_models.keys())

        for k,v in d.items():
            self.out_sockets[k].set_determine_placeholders(v)

    def infer_type_models(self, mode):
        in_types = {}
        for k, in_socket in self.in_sockets.items():
            if self.execution_type.is_used(k, mode):
                in_types[k] = in_socket.pull_type_model(mode)

        self.output_type_models = self.execution_type.build_value_type_model(in_types, self.value_model, mode)

        for k,v in self.output_type_models.items():
            self.out_sockets[k].set_cached_type(v)

    def count_parameters(self):
        params = self.value_model.count_parameters()
        if params > 0:
            message = " * " + self.get_name() + ": " + str(params)
            context = "training"
            field = "parameters"
            self.log(message, context, field)
        return params

    def get_name(self):
        return self.name

    def get_referenced_components(self):
        cs = []
        for referenced_graph in self.get_referenced_graphs():
            cs.extend(referenced_graph.get_components())
        return cs

    def get_referenced_graphs(self):
        return self.value_model.get_referenced_graphs()

    def get_in_sockets(self):
        return list(self.in_sockets.values())

    def get_out_sockets(self):
        return list(self.out_sockets.values())

    def clear_caches(self):
        self.cached_has_batches = None
        for in_socket in self.get_in_sockets():
            in_socket.clear_caches()

    cached_has_batches = None

    def has_batches(self, mode):
        if self.cached_has_batches is None:
            in_batches = {k:v.has_batches(mode) for k,v in self.in_sockets.items()}
            self.cached_has_batches = self.execution_type.has_batches(self.value_model, in_batches, mode)

        return self.cached_has_batches

    def describe_graph(self, indent=0):
        print("\t"*indent + self.get_name() + " (" + self.execution_type.name + ")")

        for in_socket in self.get_in_sockets():
            in_socket.describe_graph(indent=indent+1)

    def init_batches(self):
        for in_socket in self.get_in_sockets():
            in_socket.init_batches()
        self.value_model.init_batches()

    def __str__(self):
        return "Unknown" if self.execution_type is None else self.execution_type.name

    def get_past(self):
        past = [self]

        for socket in self.get_in_sockets():
            past.extend(socket.get_past())

        return list(set(past))

    def get_regularization(self, mode="train"):
        return self.execution_type.compute_regularization(self, mode=mode)

    def past_regularization(self, mode="train"):
        reg = 0
        for component in self.get_past():
            reg += component.get_regularization(mode=mode)

        return reg

    def has_referenced_graphs(self, mode):
        return self.execution_type.has_referenced_graphs(self.value_model, mode)

    def get_referenced_sockets(self, mode):
        return self.value_model.get_referenced_sockets(mode)