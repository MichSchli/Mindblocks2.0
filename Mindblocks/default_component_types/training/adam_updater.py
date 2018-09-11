from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.special.special_type_model import SpecialTypeModel


class AdamUpdater(ComponentTypeModel):

    name = "AdamUpdater"
    in_sockets = ["loss"]
    out_sockets = ["update"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = AdamUpdaterValue()

        if "learning_rate" in value_dictionary:
            value.set_learning_rate(float(value_dictionary["learning_rate"][0][0]))

        if "gradient_clip" in value_dictionary:
            value.set_gradient_clip(float(value_dictionary["gradient_clip"][0][0]))

        return value

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        per_example_loss = input_dictionary["loss"].get_value()
        reg = execution_component.past_regularization(mode=mode)

        per_batch_loss = tf.reduce_mean(per_example_loss)
        total_loss = per_batch_loss + reg

        adam = tf.train.AdamOptimizer(learning_rate=value.learning_rate)
        grad_and_var_pairs = adam.compute_gradients(total_loss)

        grads = [gvp[0] for gvp in grad_and_var_pairs]
        tvars = [gvp[1] for gvp in grad_and_var_pairs]

        grads, _ = tf.clip_by_global_norm(grads, value.gradient_clip)
        update = adam.apply_gradients(zip(grads, tvars))

        output_value_models["update"].assign(update)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        return {"update": SpecialTypeModel()}

class AdamUpdaterValue(ExecutionComponentValueModel):

    learning_rate = None
    gradient_clip = None

    def __init__(self):
        self.learning_rate = 0.001
        self.gradient_clip = 1

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    def set_gradient_clip(self, clip):
        self.gradient_clip = clip