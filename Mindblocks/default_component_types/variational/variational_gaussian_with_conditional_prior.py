from tensorflow_probability.python.mcmc.slice_sampler_utils import tfd

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import tensorflow_probability as tfp

class VariationalGaussianConditionalPrior(ComponentTypeModel):

    name = "VariationalGaussianConditionalPrior"
    in_sockets = ["input", "prior_input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = VariationalGaussianConditionalPriorValue()

        if "kl_scaling" in value_dictionary:
            value.set_kl_scaling(float(value_dictionary["kl_scaling"][0][0]))

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if mode != "train":
            mu, sigma = tf.split(input_dictionary["prior_input"].get_value(), 2, axis=-1)
            value.set_prior(mu, sigma)
            output = value.get_prior().sample()
            output_models["output"].assign(output)
        else:
            mu, sigma = tf.split(input_dictionary["input"].get_value(), 2, axis=-1)
            prior_mu, prior_sigma = tf.split(input_dictionary["prior_input"].get_value(), 2, axis=-1)
            value.set_posterior(mu, sigma)
            value.set_prior(prior_mu, prior_sigma)
            encoder = value.get_posterior().sample()
            output_models["output"].assign(encoder)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        inner_dim = input_types["prior_input"].get_inner_dim() / 2
        value.set_dim(inner_dim)
        output_type = TensorTypeModel("float", [None, value.dim])

        return {"output": output_type}

    def is_used(self, socket_name, mode):
        if socket_name == "input":
            return mode == "train"
        else:
            return True

    def compute_regularization(self, component, mode="train"):
        value = component.get_value_model()
        divergence = tfd.kl_divergence(value.get_posterior(), value.get_prior())
        divergence = tf.Print(divergence, [value.kl_scaling], message="kl scale", summarize=100)
        divergence = tf.Print(divergence, [value.kl_scaling * tf.reduce_mean(divergence)], message="total kl", summarize=100)
        return value.kl_scaling * tf.reduce_mean(divergence)


class VariationalGaussianConditionalPriorValue(ExecutionComponentValueModel):

    kl_scaling = None
    increase_per_iteration = 0.1

    def __init__(self):
        self.prior = None
        self.posterior = None

    def set_kl_scaling(self, factor):
        self.kl_scaling = factor

    def set_prior(self, mu, sigma):
        self.prior = tfp.distributions.MultivariateNormalDiag(mu, sigma)

    def set_posterior(self, mu, sigma):
        self.posterior = tfp.distributions.MultivariateNormalDiag(mu, sigma)

    def get_prior(self):
        return self.prior

    def get_posterior(self):
        return self.posterior

    def initialize_tensorflow_variables(self, tensorflow_session_model):
        kl_scaling = tf.minimum(1.0, self.kl_scaling + self.increase_per_iteration * tf.cast(tensorflow_session_model.get_tensorflow_iteration(), dtype=tf.float32))
        self.kl_scaling = kl_scaling

    def set_dim(self, dim):
        self.dim = dim