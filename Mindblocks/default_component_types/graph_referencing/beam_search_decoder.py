from tensorflow.python.ops import tensor_array_ops

from Mindblocks.default_component_types.graph_referencing.rnn_helper.rnn_helper import RnnHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import tensorflow as tf
from tensorflow.python.ops import math_ops

class BeamSearchDecoderComponent(ComponentTypeModel):

    name = "BeamSearchDecoder"
    in_sockets = []
    out_sockets = ["predictions"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        rnn_helper = RnnHelper()
        value = BeamSearchDecoderComponentValue()
        value.rnn_model = rnn_helper.create_rnn_model(value_dictionary)

        #Add recurrency along the beam
        beam_input = value_dictionary["beam"][0]
        parts = beam_input[0].split("->")
        init = beam_input[1]["init"] if "init" in beam_input[1] else "zero_tensor:|int"
        value.rnn_model.add_recurrence(parts[0], parts[1], init=init)

        return value

    def execute(self, input_dictionary, value, output_models, mode):
        outputs, lengths = value.assign_and_run(input_dictionary)

        for k,v in outputs.items():
            if output_models[k].is_value_type("sequence"):
                output_models[k].assign_with_lengths(v, lengths)
            else:
                output_models[k].assign(v, language="tensorflow")

        return output_models

    def build_value_type_model(self, input_types, value):
        rnn_helper = RnnHelper()
        rnn_helper.tile_batches(value.rnn_model, value.beam_width)
        rnn_helper.handle_input_types(value.rnn_model, input_types)

        #value.assign_input_types(input_types)
        output_types = value.compute_types()

        return output_types


class BeamSearchDecoderComponentValue:

    rnn_model = None
    beam_index = 2
    stop_symbol = 8
    beam_width = 3

    maximum_iterations = 100

    vocab_size = 9

    def set_graph(self, graph):
        self.rnn_model.set_inner_graph(graph)

    def body(self, *args):
        n_rec = self.rnn_model.count_recurrent_links()
        for i in range(n_rec):
            self.rnn_model.set_nths_input(i, args[i])

        results = self.rnn_model.run()

        print(results)

        beam_scores = results[self.beam_index]
        reshaped_beam_scores = tf.reshape(beam_scores, [self.rnn_model.batch_size, -1])

        next_beam_scores, next_beam_indices = tf.nn.top_k(reshaped_beam_scores, k=self.beam_width)

        next_word_ids = math_ops.to_int32(next_beam_indices % self.vocab_size, name="next_beam_word_ids")
        next_beam_ids = math_ops.to_int32(next_beam_indices / self.vocab_size, name="next_beam_parent_ids")

        combined_indices = tf.reshape(tf.stack([next_beam_ids, next_word_ids], -1), [-1, 2])

        # Select recurrent states from the appropriate beam
        range_ = tf.expand_dims(math_ops.range(self.rnn_model.batch_size) * self.beam_width, 1)
        gather_indices = tf.reshape(next_beam_ids + range_, [-1])

        for i in range(n_rec):
            if i != self.beam_index:
                results[i] = tf.gather(results[i], indices=tf.reshape(gather_indices, [-1]))
            else:
                results[i] = 

        print(results)

        print(reshaped_beam_scores)
        print(next_beam_ids)
        print(next_word_ids)

        print(next_beam_indices)

        exit()
        n_out = self.rnn_model.count_output_links()

        counter = args[-1]

        #for i in range(n_rec, n_rec+n_out):
        #    results[i] = self.write_to_tensor_array(args[i],results[i], counter)

        should_stop_prediction = tf.equal(results[self.teacher_index], self.stop_symbol)
        old_finished = args[-2]
        new_finished = tf.logical_or(old_finished, should_stop_prediction)

        old_lengths = args[-3]
        new_lengths = tf.where(tf.logical_not(old_finished), old_lengths + 1, old_lengths)

        results += [new_lengths]
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
        self.rnn_model.loop_vars = []
        rnn_helper = RnnHelper()
        rnn_helper.assign_static_inputs(self.rnn_model, input_dictionary)

        # TODO: Code for feeding input sequence missing;
        #for component_input, graph_input, feed_type in self.in_links:
        #    parts = graph_input.split(":")
        #    if feed_type == "loop":
        #        sequence_feeds.append(input_dictionary[component_input])
        #        sequence_sockets.append((parts[0], parts[1]))

        rnn_helper.add_recurrency_initializers(self.rnn_model, input_dictionary)

        # predictions, backpointers:
        self.rnn_model.build_loop_var("zero_tensor:|int", name="beamsearch_predictions")
        self.rnn_model.build_loop_var("zero_tensor:|int", name="beamsearch_backpointers")

        self.rnn_model.add_length_var()
        self.rnn_model.add_finished_var()
        self.rnn_model.add_counter_loop_var()

        loop = tf.while_loop(
            self.cond,
            self.body,
            loop_vars=tuple(self.rnn_model.loop_vars),
            maximum_iterations=self.maximum_iterations
        )
        loop = list(loop)

        return self.order_output(loop)

    def order_output(self, loop):
        lengths = loop[-3]

        n_rec = self.rnn_model.count_recurrent_links()
        n_out = self.rnn_model.count_output_links()
        for i in range(n_rec, n_rec + n_out):
            loop[i] = tf.transpose(loop[i].stack(), perm=[1,0,2])
        return {self.rnn_model.out_links[i][0]: loop[i + n_rec] for i in range(len(self.rnn_model.out_links))}, lengths

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