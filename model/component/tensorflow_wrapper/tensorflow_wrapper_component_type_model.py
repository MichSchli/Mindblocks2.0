from model.component.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

class TensorflowWrapperComponentTypeModel(ComponentTypeModel):

    name = "__tensorflow_wrapper__"

    def execute(self, in_sockets, value, language="python"):
        if value.graph is None:
            value.compile_graph()

        feed_dict = {}
        for i in range(len(in_sockets)):
            feed_dict[value.get_variables()[i]] = in_sockets[i]

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            output = sess.run(value.graph, feed_dict)

        return output

    def evaluate_value_type(self, in_types, value):
        return value.get_out_value_types()