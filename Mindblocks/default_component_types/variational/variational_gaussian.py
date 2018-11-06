import tensorflow as tf
import tensorflow_probability as tfp

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class VariationalGaussian(ComponentTypeModel):

    name = "VariationalGaussian"
    in_sockets = ["input", "test_input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = VariationalGaussianValue()

        if "kl_scaling" in value_dictionary:
            value.set_kl_scaling(float(value_dictionary["kl_scaling"][0][0]))

        if "kl_annealing" in value_dictionary:
            value.set_kl_annealing(float(value_dictionary["kl_annealing"][0][0]))

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if mode != "train":
            batch_size = tf.shape(input_dictionary["test_input"].get_value())[0]
            output = value.get_prior().sample(batch_size)
            output_models["output"].assign(output, length_list=None)
        else:
            mu, sigma = tf.split(input_dictionary["input"].get_value(), 2, axis=-1)
            sigma = tf.nn.softplus(sigma)
            value.set_posterior(mu, sigma)
            encoder = value.get_posterior().sample()
            output_models["output"].assign(encoder, length_list=None)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        if value.prior is None:
            if mode == "train":
                inner_dim = input_types["input"].get_dimension(-1) / 2
            else:
                inner_dim = input_types["test_input"].get_dimension(-1) / 2

            value.initialize_prior(inner_dim)

        if mode == "train":
            batch_size = input_types["input"].get_dimension(0)
        else:
            batch_size = input_types["test_input"].get_dimension(0)

        output_type = SoftTensorTypeModel([batch_size, value.dim], string_type="float")

        return {"output": output_type}

    def is_used(self, socket_name, mode):
        if socket_name == "input":
            return mode == "train"
        elif socket_name == "test_input":
            return mode != "train"

    def compute_regularization(self, component, mode="train"):
        value = component.get_value_model()
        divergence = tfp.distributions.kl_divergence(value.get_posterior(), value.get_prior())
        divergence = tf.Print(divergence, [value.kl_scaling], message="kl scale", summarize=100)
        divergence = tf.Print(divergence, [value.kl_scaling * tf.reduce_mean(divergence)], message="total kl", summarize=100)
        return value.kl_scaling * tf.reduce_mean(divergence)


class VariationalGaussianValue(ExecutionComponentValueModel):

    kl_scaling = None
    increase_per_iteration = 0.1

    def __init__(self):
        self.prior = None
        self.posterior = None

    def set_kl_scaling(self, factor):
        self.kl_scaling = factor

    def set_posterior(self, mu, sigma):
        self.posterior = tfp.distributions.MultivariateNormalDiag(mu, sigma)

    def get_prior(self):
        return self.prior

    def get_posterior(self):
        return self.posterior

    def set_kl_annealing(self, increase_per_iteration):
        self.increase_per_iteration = increase_per_iteration

    def initialize_tensorflow_variables(self, tensorflow_session_model):
        kl_scaling = tf.minimum(1.0, self.kl_scaling + self.increase_per_iteration * tf.cast(tensorflow_session_model.get_tensorflow_iteration(), dtype=tf.float32))
        self.kl_scaling = kl_scaling

    def initialize_prior(self, dim):
        self.dim = dim

        mu = tf.zeros(self.dim)
        sigma = tf.ones(self.dim)
        self.prior = tfp.distributions.MultivariateNormalDiag(mu, sigma)