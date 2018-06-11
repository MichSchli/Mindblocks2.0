from model.component.component_type.component_type_model import ComponentTypeModel
import tensorflow as tf

class TensorflowWrapperComponentTypeModel(ComponentTypeModel):

    name = "__tensorflow_wrapper__"

    def execute(self, in_sockets, value):
        feed_dict = {}
        for i in range(len(in_sockets)):
            feed_dict[value.inner_component.in_sockets[i]] = in_sockets[i]

        with tf.Session() as sess:
            output = sess.run(value.inner_component.graph, feed_dict)

        return output