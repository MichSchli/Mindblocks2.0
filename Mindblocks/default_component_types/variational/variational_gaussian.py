from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import tensorflow_probability as tfp

class VariationalGaussian(ComponentTypeModel):

    name = "VariationalGaussian"
    in_sockets = ["input", "test_input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return VariationalGaussianValue()

    def execute(self, input_dictionary, value, output_models, mode):
        if mode != "train":
            batch_size = tf.shape(input_dictionary["test_input"].get_value())[0]
            output = value.prior.sample(batch_size)
            output_models["output"].assign(output)
        else:
            mu, sigma = tf.split(input_dictionary["input"].get_value(), 2, axis=-1)
            encoder = tfp.distributions.MultivariateNormalDiag(mu, sigma).sample()
            output_models["output"].assign(encoder)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        if value.prior is None:
            if mode == "train":
                inner_dim = input_types["train"].get_inner_dim()
            else:
                inner_dim = input_types["test_input"].get_inner_dim()

            value.initialize_prior(inner_dim)

        output_type = TensorTypeModel("float", [None, value.dim])

        return {"output": output_type}

    def is_used(self, socket_name, value, mode):
        if socket_name == "input":
            return mode == "train"
        elif socket_name == "test_input":
            return mode != "train"


class VariationalGaussianValue(ExecutionComponentValueModel):

    def __init__(self):
        self.prior = None

    def initialize_prior(self, dim):
        self.dim = dim

        mu = tf.zeros(self.dim)
        sigma = tf.ones(self.dim)
        self.prior = tfp.distributions.MultivariateNormalDiag(mu, sigma)