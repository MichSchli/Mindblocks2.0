from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.special.special_type_model import SpecialTypeModel


class SGDUpdater(ComponentTypeModel):

    name = "SGDUpdater"
    in_sockets = ["loss"]
    out_sockets = ["update"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = SGDUpdaterValue()

        if "learning_rate" in value_dictionary:
            value.set_learning_rate(float(value_dictionary["learning_rate"][0][0]))

        if "gradient_clip" in value_dictionary:
            value.set_gradient_clip(float(value_dictionary["gradient_clip"][0][0]))

        if "learning_rate_decay" in value_dictionary:
            if "start_iteration" in value_dictionary["learning_rate_decay"][0][1]:
                start_iteration = int(value_dictionary["learning_rate_decay"][0][1]["start_iteration"])
            else:
                start_iteration = None

            value.set_learning_rate_decay(float(value_dictionary["learning_rate_decay"][0][0]), start_iteration=start_iteration)

        return value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        per_example_loss = input_dictionary["loss"].get_value()
        per_batch_loss = tf.reduce_mean(per_example_loss)
        optim = tf.train.GradientDescentOptimizer(learning_rate=value.learning_rate)
        grad_and_var_pairs = optim.compute_gradients(per_batch_loss)

        grads = [gvp[0] for gvp in grad_and_var_pairs]
        tvars = [gvp[1] for gvp in grad_and_var_pairs]

        if value.gradient_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, value.gradient_clip)
        update = optim.apply_gradients(zip(grads, tvars))

        output_value_models["update"].assign(update)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"update": SpecialTypeModel()}

class SGDUpdaterValue(ExecutionComponentValueModel):

    learning_rate = None
    gradient_clip = None

    decay_rate = None
    decay_start_iteration = None

    def __init__(self):
        self.learning_rate = 0.001
        self.gradient_clip = None

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    def set_gradient_clip(self, clip):
        self.gradient_clip = clip

    def set_learning_rate_decay(self, decay_rate, start_iteration=None):
        self.decay_rate = decay_rate
        self.decay_start_iteration = start_iteration

    def get_learning_rate(self):
        return self.learning_rate

    def initialize_tensorflow_variables(self, tensorflow_session_model):
        lr = tf.Variable(self.learning_rate, trainable=False)
        if self.decay_rate is not None:
            if self.decay_start_iteration is not None:
                iteration = tensorflow_session_model.get_tensorflow_iteration()
                lr = tf.cond(iteration >= self.decay_start_iteration,
                             lambda: lr * self.decay_rate ** tf.cast((iteration - self.decay_start_iteration + 1), dtype=tf.float32),
                             lambda: lr)
            else:
                lr = lr * self.decay_rate ** tensorflow_session_model.get_tensorflow_iteration()

        self.learning_rate = lr