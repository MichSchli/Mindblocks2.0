from tensorflow.python.ops import tensor_array_ops

from Mindblocks.default_component_types.graph_referencing.rnn_helper.rnn_helper import RnnHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import tensorflow as tf
from tensorflow.python.ops import math_ops

class ScheduledSamplingRnnComponent(ComponentTypeModel):

    name = "ScheduledSamplingRnn"
    in_sockets = ["teacher_inputs"]
    out_sockets = []
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        rnn_helper = RnnHelper()
        value = ScheduledSamplingRnnComponentValue()
        value.rnn_model = rnn_helper.create_rnn_model(value_dictionary)

        # Find the teacher:
        counter = 0
        for recurrence in value_dictionary["recurrence"]:
            print(recurrence[1])
            if "teacher" in recurrence[1]:
                print("abc")
                value.set_teacher_index(counter)

            counter += 1

        return value

    def execute(self, input_dictionary, value, output_models, mode):
        print(mode)
        outputs = value.assign_and_run(input_dictionary)

        for k,v in outputs.items():
            output_models[k].assign(v, language="tensorflow")

        print(output_models)

        return output_models

    def build_value_type_model(self, input_types, value):
        batch_size = input_types["teacher_inputs"].get_batch_size()
        value.rnn_model.set_batch_size(batch_size)

        rnn_helper = RnnHelper()
        rnn_helper.handle_input_types(value.rnn_model, input_types)

        #value.assign_input_types(input_types)
        output_types = value.compute_types()

        return output_types


class ScheduledSamplingRnnComponentValue:

    rnn_model = None
    teacher_index = 0

    initial_teacher_probability = 0.9
    final_teacher_probability = 0.5
    decay_rate = 0.999

    stop_symbol = 8

    def __init__(self):
        self.teacher_probability = tf.Variable(initial_value=self.initial_teacher_probability, trainable=False)

    def set_teacher_index(self, index):
        self.teacher_index = index

    def set_graph(self, graph):
        self.rnn_model.set_inner_graph(graph)

    def body(self, *args):
        n_rec = self.rnn_model.count_recurrent_links()
        n_out = self.rnn_model.count_output_links()

        for i in range(n_rec):
            if i == self.teacher_index:
                next_teacher_value = self.get_teacher_value(args[-1])
                student_suggestion = args[self.teacher_index]

                coin_flip = tf.random_uniform([self.rnn_model.batch_size], minval=0, maxval=1)
                chosen_input = tf.where(coin_flip < self.teacher_probability, x=next_teacher_value, y=student_suggestion)

                self.rnn_model.set_nths_input(i, chosen_input)
            else:
                self.rnn_model.set_nths_input(i, args[i])

        results = self.rnn_model.run()

        counter = args[-1]
        for i in range(n_rec, n_rec+n_out):
            results[i] = self.write_to_tensor_array(args[i],results[i], counter)

        should_stop = tf.equal(results[self.teacher_index], self.stop_symbol)
        new_finished = tf.logical_or(args[-2], should_stop)
        new_finished = tf.Print(new_finished, [self.get_teacher_value(args[-1])], summarize=100, message="teach ")
        new_finished = tf.Print(new_finished, [results[self.teacher_index]], summarize=100, message="pred ")
        new_finished = tf.Print(new_finished, [should_stop], summarize=100, message="should ")

        #TODO: fix lengths

        results += [new_finished]
        results += [counter + 1]

        return tuple(results)

    def write_to_tensor_array(self, array, item, index):
        return array.write(index, item)

    def cond(self, *args):
        finished = args[-2]
        return math_ops.logical_not(math_ops.reduce_all(finished))

    def get_teacher_value(self, index):
        return self.teacher_values[index]

    def assign_and_run(self, input_dictionary):
        self.teacher_probability = tf.assign(self.teacher_probability,
                                        value=tf.reduce_max([self.teacher_probability * self.decay_rate,
                                                             self.final_teacher_probability]))
        self.rnn_model.loop_vars = []
        rnn_helper = RnnHelper()
        rnn_helper.assign_static_inputs(self.rnn_model, input_dictionary)

        maximum_iterations = input_dictionary["teacher_inputs"].get_maximum_sequence_length()

        self.teacher_values = tf.transpose(input_dictionary["teacher_inputs"].get_sequences(), perm=[1,0])

        # TODO: Code for feeding input sequence missing;
        #for component_input, graph_input, feed_type in self.in_links:
        #    parts = graph_input.split(":")
        #    if feed_type == "loop":
        #        sequence_feeds.append(input_dictionary[component_input])
        #        sequence_sockets.append((parts[0], parts[1]))

        rnn_helper.add_recurrency_initializers(self.rnn_model, input_dictionary)
        rnn_helper.add_sequence_outputs(self.rnn_model, maximum_iterations)

        self.rnn_model.add_finished_var()
        self.rnn_model.add_counter_loop_var()

        loop = tf.while_loop(
            self.cond,
            self.body,
            loop_vars=tuple(self.rnn_model.loop_vars),
            maximum_iterations=maximum_iterations
        )
        loop = list(loop)

        return self.order_output(loop, maximum_iterations)

    def order_output(self, loop, maximum_iterations):
        n_rec = self.rnn_model.count_recurrent_links()
        n_out = self.rnn_model.count_output_links()
        for i in range(n_rec, n_rec + n_out):
            print(loop[i])
            loop[i] = tf.transpose(loop[i].stack(), perm=[1,0,2])
            #loop[i] = tf.reshape(loop[i].stack(), [self.rnn_model.batch_size, maximum_iterations, -1])
        return {self.rnn_model.out_links[i][0]: loop[i + n_rec] for i in range(len(self.rnn_model.out_links))}

    def count_recurrences(self):
        return len(self.recurrences)

    def run_graph(self):
        results = self.graph.execute()
        return {output[0]: result for output, result in zip(self.out_links, results)}

    def compute_types(self):
        inner_graph_output = self.rnn_model.get_inner_graph_output_types()

        return inner_graph_output

    def set_graph_name(self, name):
        self.graph_name = name

    def get_populate_items(self):
        return [("graph", {"name": self.rnn_model.get_graph_name()})]

    def get_required_graph_outputs(self):
        return self.rnn_model.get_required_graph_outputs()

    def get_graph_inputs(self):
        return self.rnn_model.get_required_graph_inputs()