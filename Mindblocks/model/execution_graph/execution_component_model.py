from Mindblocks.helpers.logging.logger_factory import LoggerFactory
from Mindblocks.model.abstract.abstract_model import AbstractModel


class ExecutionComponentModel(AbstractModel):

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

    def add_out_socket(self, key, socket):
        self.out_sockets[key] = socket

    def add_in_socket(self, key, socket):
        self.in_sockets[key] = socket

    def execute(self, mode):
        input_dictionary = {k : in_socket.pull(mode) for k,in_socket in self.in_sockets.items()}
        output_value_models = {k : type_model.initialize_value_model() for k,type_model in self.output_type_models.items()}

        output_dictionary = self.execution_type.execute(input_dictionary, self.execution_value, output_value_models, mode)

        for k,v in output_dictionary.items():
            self.out_sockets[k].set_cached_value(v)

    def initialize(self, mode, tensorflow_session_model):
        input_dictionary = {k: in_socket.initialize(mode, tensorflow_session_model) for k, in_socket in self.in_sockets.items()}
        output_value_models = {k: type_model.initialize_value_model() for k, type_model in
                               self.output_type_models.items()}

        output_dictionary = self.execution_type.initialize(input_dictionary,
                                                           self.execution_value,
                                                           output_value_models,
                                                           tensorflow_session_model)

        for k,v in output_dictionary.items():
            self.out_sockets[k].set_cached_init_value(v)

    def determine_placeholders(self):
        d = self.execution_type.determine_placeholders(self.execution_value, self.output_type_models.keys())

        for k,v in d.items():
            self.out_sockets[k].set_determine_placeholders(v)

    def infer_type_models(self, mode):
        in_types = {}
        for k, in_socket in self.in_sockets.items():
            if self.execution_type.is_used(k, self.execution_value, mode):
                in_types[k] = in_socket.pull_type_model(mode)

        self.output_type_models = self.execution_type.build_value_type_model(in_types, self.execution_value, mode)

        for k,v in self.output_type_models.items():
            self.out_sockets[k].set_cached_type(v)

    def count_parameters(self):
        params = self.execution_value.count_parameters()
        if params > 0:
            message = " * " + self.get_name() + ": " + str(params)
            context = "training"
            field = "parameters"
            self.log(message, context, field)
        return params

    def get_name(self):
        return self.name

    def get_value(self):
        return self.execution_value

    def get_referenced_graphs(self):
        return self.execution_value.get_referenced_graphs()

    def get_in_sockets(self):
        return list(self.in_sockets.values())

    def get_out_sockets(self):
        return list(self.out_sockets.values())

    def clear_caches(self):
        self.cached_has_batches = None
        for in_socket in self.get_in_sockets():
            in_socket.clear_caches()

    cached_has_batches = None

    def has_batches(self):
        if self.cached_has_batches is None:
            in_batches = {k:v.has_batches() for k,v in self.in_sockets.items()}
            self.cached_has_batches = self.execution_type.has_batches(self.execution_value, in_batches)

        return self.cached_has_batches

    def describe_graph(self, indent=0):
        print("\t"*indent + self.execution_type.name)

        for in_socket in self.get_in_sockets():
            in_socket.describe_graph(indent=indent+1)

    def init_batches(self):
        for in_socket in self.get_in_sockets():
            in_socket.init_batches()
        self.execution_value.init_batches()

    def __str__(self):
        return "Unknown" if self.execution_type is None else self.execution_type.name