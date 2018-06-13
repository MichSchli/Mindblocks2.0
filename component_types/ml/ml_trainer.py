from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel
import tensorflow as tf


class MlTrainer(ComponentTypeModel):

    name = "MlTrainer"
    in_socket_names = ["loss"]
    out_socket_names = ["gradient_update", "loss"]
    available_languages = ["tensorflow"]

    def __init__(self):
        pass

    def get_new_value(self):
        return MlTrainerValue()

    def execute(self, in_sockets, value, language="tensorflow"):
        parameters_to_optimize = tf.trainable_variables()
        opt_func = tf.train.AdamOptimizer(learning_rate=value.learning_rate)
        gradients = tf.gradients(in_sockets[0], parameters_to_optimize)
        optimize_func = opt_func.apply_gradients(zip(gradients, parameters_to_optimize))

        return [optimize_func, in_sockets[0]]

class MlTrainerValue(ComponentValueModel):

    learning_rate = None

    def __init__(self):
        self.learning_rate = 0.001

    def load(self, value_lines):
        pass

    def copy(self):
        copy = MlTrainerValue()
        copy.learning_rate = self.learning_rate
        return copy

    def describe(self):
        return "abc"
