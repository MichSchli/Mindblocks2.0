from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class AdamUpdater(ComponentTypeModel):

    name = "AdamUpdater"
    in_sockets = ["loss"]
    out_sockets = ["update"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return AdamUpdaterValue()

    def execute(self, input_dictionary, value, mode):
        adam = tf.train.AdamOptimizer(learning_rate=value.learning_rate)
        grad_and_var_pairs = adam.compute_gradients(input_dictionary["loss"])
        update = adam.apply_gradients(grad_and_var_pairs)
        return {"update": update}

    def infer_types(self, input_types, value):
        return {"update": input_types["input"]}

    def infer_dims(self, input_dims, value):
        return {"update": input_dims["input"]}

class AdamUpdaterValue(ExecutionComponentValueModel):

    learning_rate = 0.001