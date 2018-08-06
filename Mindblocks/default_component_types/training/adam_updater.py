from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.special.special_type_model import SpecialTypeModel


class AdamUpdater(ComponentTypeModel):

    name = "AdamUpdater"
    in_sockets = ["loss"]
    out_sockets = ["update"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return AdamUpdaterValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        adam = tf.train.AdamOptimizer(learning_rate=value.learning_rate)
        grad_and_var_pairs = adam.compute_gradients(input_dictionary["loss"].get_value())
        update = adam.apply_gradients(grad_and_var_pairs)
        output_value_models["update"].assign(update)
        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"update": SpecialTypeModel()}

class AdamUpdaterValue(ExecutionComponentValueModel):

    learning_rate = 0.001