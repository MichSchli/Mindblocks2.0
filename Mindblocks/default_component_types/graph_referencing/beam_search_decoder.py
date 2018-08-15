from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops, array_ops

from Mindblocks.default_component_types.graph_referencing.rnn_helper.rnn_helper import RnnHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
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
        decoded_sequences, lengths = value.assign_and_run(input_dictionary)

        output_models["predictions"].assign_with_lengths(decoded_sequences, lengths)

        return output_models

    def build_value_type_model(self, input_types, value):
        rnn_helper = RnnHelper()
        rnn_helper.tile_batches(value.rnn_model, value.beam_width)
        rnn_helper.handle_input_types(value.rnn_model, input_types)

        #value.assign_input_types(input_types)
        output_types = value.compute_types()

        # TODO this is nonsence
        output_types["predictions"] = SequenceBatchTypeModel("int", [], 30, 30)

        return output_types


class BeamSearchDecoderComponentValue:

    rnn_model = None
    beam_index = 2
    stop_symbol = 8
    beam_width = 3

    maximum_iterations = 10

    vocab_size = 9

    length_penalty = 0.6 # Check google nmt: 0.6-0/7 best

    def set_graph(self, graph):
        self.rnn_model.set_inner_graph(graph)

    def calculate_word_and_beam_pointers(self, scores):
        reshaped_beam_scores = tf.reshape(scores, [self.rnn_model.batch_size, -1])
        next_beam_scores, next_beam_indices = tf.nn.top_k(reshaped_beam_scores, k=self.beam_width)
        next_word_ids = math_ops.to_int32(next_beam_indices % self.vocab_size, name="next_beam_word_ids")
        next_beam_ids = math_ops.to_int32(next_beam_indices / self.vocab_size, name="next_beam_parent_ids")
        return next_beam_ids, next_beam_scores, next_word_ids

    def calculate_initial_iteration_word_and_beam_pointers(self, scores):
        batch_beam_scores = tf.reshape(scores, [self.rnn_model.batch_size, self.beam_width, -1])
        first_beam_scores = batch_beam_scores[:, 0, :]

        next_beam_scores, next_word_ids = tf.nn.top_k(first_beam_scores, k=self.beam_width)
        next_beam_ids = tf.zeros([self.rnn_model.batch_size, self.beam_width], dtype=tf.int32)

        return next_beam_ids, next_beam_scores, next_word_ids

    def body(self, *args):
        counter = args[-1]
        old_finished = args[-2]
        n_rec = self.rnn_model.count_recurrent_links()
        score_index = n_rec + 2

        for i in range(n_rec):
            self.rnn_model.set_nths_input(i, args[i])

        results = self.rnn_model.run()

        # Calculate scores:
        scores = tf.nn.log_softmax(results[self.beam_index])

        # Overwrite with old scores for finished beams:
        tf_type = tf.float32
        log_prob_for_finished = tf.expand_dims(
            array_ops.one_hot(self.stop_symbol,
                              self.vocab_size,
                              dtype=tf_type,
                              on_value=ops.convert_to_tensor(0., dtype=tf_type),
                              off_value=tf_type.min), 0)
        finished_mask = tf.tile(log_prob_for_finished, [self.rnn_model.batch_size*self.beam_width, 1])
        scores = tf.where(old_finished, finished_mask, scores)

        # Add old scores to accumulate:
        scores += tf.expand_dims(args[score_index], -1)

        # Update lengths:
        old_lengths = args[-3]
        new_lengths = tf.where(tf.logical_not(old_finished), old_lengths + 1, old_lengths)

        # Compute length penalty:
        if self.length_penalty is not None:
            length_penalty = math_ops.div((5. + math_ops.to_float(new_lengths)) ** self.length_penalty, (5. + 1.) ** self.length_penalty)
            scores /= tf.expand_dims(length_penalty, -1)

        # Use old scores for finished beams
        parent_beam_ids, next_beam_scores, next_word_ids = tf.cond(tf.equal(counter,0),
                                                                 lambda: self.calculate_initial_iteration_word_and_beam_pointers(scores),
                                                                 lambda: self.calculate_word_and_beam_pointers(scores))

        parent_beam_ids = tf.Print(parent_beam_ids, [parent_beam_ids], summarize=100, message="next beam ids")
        next_word_ids = tf.Print(next_word_ids, [next_word_ids], summarize=100, message="next word ids")

        # Select recurrent states from the appropriate beam
        range_ = tf.expand_dims(math_ops.range(self.rnn_model.batch_size) * self.beam_width, 1)
        gather_indices = tf.reshape(parent_beam_ids + range_, [-1])
        gather_indices = tf.reshape(gather_indices, [-1])

        # Determine which of the current beams are continuations of finished beams. Replace tokens with stop tokens:
        continues_finished = tf.gather(old_finished, indices=gather_indices)
        next_word_ids = tf.where(tf.reshape(continues_finished, [-1, self.beam_width]), tf.ones_like(next_word_ids) * self.stop_symbol, next_word_ids)

        # Add newly finished beams to list of finished:
        should_stop_prediction = tf.equal(tf.reshape(next_word_ids, [-1]), self.stop_symbol)
        new_finished = tf.logical_or(continues_finished, should_stop_prediction)

        # Allocate lengths to beams:
        new_lengths = tf.gather(new_lengths, indices=gather_indices)

        # Distribute reccurent values according to beams:
        for i in range(n_rec):
            if i != self.beam_index:
                results[i] = tf.gather(results[i], indices=gather_indices)
            else:
                results[i] = tf.reshape(next_word_ids, [-1])

        # Store backpointers:
        prediction_index = n_rec
        backpointer_index = n_rec + 1
        results.append(self.write_to_tensor_array(args[prediction_index], tf.reshape(next_word_ids, [-1]), counter))
        results.append(self.write_to_tensor_array(args[backpointer_index], tf.reshape(parent_beam_ids, [-1]), counter))

        results.append(tf.reshape(next_beam_scores, [-1]))

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
        self.rnn_model.build_tensor_array_loop_var("zero_tensor:|int", name="beamsearch_predictions", maximum_iterations=self.maximum_iterations)
        self.rnn_model.build_tensor_array_loop_var("zero_tensor:|int", name="beamsearch_backpointers", maximum_iterations=self.maximum_iterations)
        self.rnn_model.build_loop_var("zero_tensor:|float", name="beamsearch_propagated_scores")

        self.rnn_model.add_length_var()
        self.rnn_model.add_finished_var()
        self.rnn_model.add_counter_loop_var()

        loop = tf.while_loop(
            self.cond,
            self.body,
            loop_vars=tuple(self.rnn_model.loop_vars),
            maximum_iterations=self.maximum_iterations
        )

        n_rec = self.rnn_model.count_recurrent_links()
        prediction_index = n_rec
        backpointer_index = n_rec + 1
        score_index = n_rec + 2

        predictions = tf.reshape(loop[prediction_index].stack(), [-1, self.rnn_model.batch_size, self.beam_width])
        backpointers = tf.reshape(loop[backpointer_index].stack(), [-1, self.rnn_model.batch_size, self.beam_width])


        predictions = tf.Print(predictions, [predictions[-1]], summarize=200, message="predictions: ")
        backpointers = tf.Print(backpointers, [backpointers[-1]], summarize=200, message="backpointers: ")

        lengths = loop[-3]
        lengths = tf.Print(lengths, [lengths], summarize=100, message="length")
        decoded_sequences = tf.transpose(tf.reshape(self.decode(predictions, backpointers, tf.reshape(lengths, [-1, self.beam_width])), [-1, 30]), [1,0])

        return decoded_sequences, lengths

    def decode(self, predictions, backpointers, lengths):
        #TODO: Replace with own
        decoded_seqs = tf.contrib.seq2seq.gather_tree(predictions,
                                                      backpointers,
                                                      tf.reduce_max(lengths, axis=-1),
                                                      self.stop_symbol)

        decoded_seqs = tf.Print(decoded_seqs, [decoded_seqs], summarize=100, message="decoded: ")

        return decoded_seqs

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