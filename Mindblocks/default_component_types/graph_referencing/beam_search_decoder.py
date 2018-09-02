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
        value.set_beam_index(value.rnn_model.count_recurrent_links())
        value.rnn_model.add_recurrence(parts[0], parts[1], init=init)

        if "n_beams" in value_dictionary:
            value.set_beam_width(int(value_dictionary["n_beams"][0][0]))

        if "output_top_n" in value_dictionary:
            value.set_n_to_output(int(value_dictionary["output_top_n"][0][0]))

        if "stop_token" in value_dictionary:
            value.set_stop_token(int(value_dictionary["stop_token"][0][0]))

        if "vocabulary_size" in value_dictionary:
            value.set_vocabulary_size(int(value_dictionary["vocabulary_size"][0][0]))

        return value

    def execute(self, input_dictionary, value, output_models, mode):
        batch_size = self.compute_batch_size(input_dictionary, value)
        value.rnn_model.set_batch_size(batch_size)
        decoded_sequences, lengths, aux_out = value.assign_and_run(input_dictionary, mode)

        output_models["predictions"].assign_with_lengths(decoded_sequences, lengths)

        for k,v in aux_out.items():
            if output_models[k].is_value_type("sequence"):
                output_models[k].assign_with_lengths(v, lengths)
            else:
                output_models[k].assign(v, language="tensorflow")

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        rnn_helper = RnnHelper()

        rnn_helper.tile_batches(value.rnn_model, value.beam_width)
        rnn_helper.handle_input_types(value.rnn_model, input_types)

        #value.assign_input_types(input_types)
        output_types = value.compute_types(mode)

        # TODO this is nonsence
        output_types["predictions"] = SequenceBatchTypeModel("int", [], None, value.maximum_iterations)

        return output_types

    def compute_batch_size(self, input_dictionary, value):
        for component_input, graph_input, feed_type in value.rnn_model.in_links:
            if feed_type == "per_batch":
                batch_size = tf.shape(input_dictionary[component_input].get_value())[0]

        return batch_size


class BeamSearchDecoderComponentValue(ExecutionComponentValueModel):

    rnn_model = None
    beam_width = None
    n_to_output = None
    stop_symbol = None
    vocab_size = None

    beam_index = None
    maximum_iterations = 50
    length_penalty = 0.6 # Check google nmt: 0.6-0/7 best

    def __init__(self):
        self.beam_width = 1
        self.n_to_output = 1
        self.stop_symbol = 0

    def get_referenced_graphs(self):
        return [self.rnn_model.inner_graph]

    def set_beam_index(self, index):
        self.beam_index = index

    def set_stop_token(self, symbol):
        self.stop_symbol = symbol

    def set_n_to_output(self, n):
        self.n_to_output = n

    def set_beam_width(self, beams):
        self.beam_width = beams

    def set_vocabulary_size(self, size):
        self.vocab_size = size

    def set_graph(self, graph):
        self.rnn_model.set_inner_graph(graph)

    def calculate_word_and_beam_pointers(self, scores):
        reshaped_beam_scores = tf.reshape(scores, [self.rnn_model.batch_size, -1])
        next_beam_scores, next_beam_indices = tf.nn.top_k(reshaped_beam_scores, k=self.beam_width)
        next_word_ids = math_ops.to_int32(next_beam_indices % self.vocab_size, name="next_beam_word_ids")
        next_beam_ids = math_ops.to_int32(next_beam_indices / self.vocab_size, name="next_beam_parent_ids")

        log_prob_indexes = next_beam_indices
        return next_beam_ids, next_beam_scores, next_word_ids, log_prob_indexes

    def calculate_initial_iteration_word_and_beam_pointers(self, scores):
        batch_beam_scores = tf.reshape(scores, [self.rnn_model.batch_size, self.beam_width, -1])
        first_beam_scores = batch_beam_scores[:, 0, :]

        next_beam_scores, next_word_ids = tf.nn.top_k(first_beam_scores, k=self.beam_width)
        next_beam_ids = tf.zeros([self.rnn_model.batch_size, self.beam_width], dtype=tf.int32)

        log_prob_indexes = next_word_ids

        return next_beam_ids, next_beam_scores, next_word_ids, log_prob_indexes

    def body(self, *args):
        counter = args[-1]
        old_finished = args[-2]
        n_rec = self.rnn_model.count_recurrent_links()
        n_out = self.rnn_model.count_output_links()
        score_index = n_rec + n_out + 2
        # TODO: Fix indexes
        # TODO: Properly return output

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
        penalized_scores = scores
        if self.length_penalty is not None:
            length_penalty = math_ops.div((5. + math_ops.to_float(new_lengths)) ** self.length_penalty, (5. + 1.) ** self.length_penalty)
            penalized_scores /= tf.expand_dims(length_penalty, -1)

        # Compute best beams:
        parent_beam_ids, next_beam_scores, next_word_ids, combined_beam_word_ids = tf.cond(tf.equal(counter,0),
                                                                                           lambda: self.calculate_initial_iteration_word_and_beam_pointers(penalized_scores),
                                                                                           lambda: self.calculate_word_and_beam_pointers(penalized_scores))

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

        # Distribute recurrent values according to beams:
        for i in range(n_rec):
            if i != self.beam_index:
                results[i] = tf.gather(results[i], indices=gather_indices)
            else:
                results[i] = tf.reshape(next_word_ids, [-1])

        # Write outputs:
        for i in range(n_rec, n_rec + n_out):
            results[i] = self.write_to_tensor_array(args[i], results[i], counter)

        # Store backpointers:
        prediction_index = n_rec + n_out
        backpointer_index = n_rec + n_out + 1
        results.append(self.write_to_tensor_array(args[prediction_index], tf.reshape(next_word_ids, [-1]), counter))
        results.append(self.write_to_tensor_array(args[backpointer_index], tf.reshape(parent_beam_ids, [-1]), counter))

        # TODO: This needs to be log prob sums
        total_log_probs = tf.reshape(scores, [self.rnn_model.batch_size * self.beam_width, -1])
        range_ = tf.expand_dims(math_ops.range(self.rnn_model.batch_size) * self.beam_width * self.vocab_size, 1)
        gather_indices = tf.reshape(combined_beam_word_ids + range_, [-1])
        gather_indices = tf.reshape(gather_indices, [-1])
        flat_log_probs = tf.reshape(total_log_probs, [-1])

        output_log_probs = tf.gather(flat_log_probs, indices=gather_indices)
        results.append(output_log_probs)

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

    def assign_and_run(self, input_dictionary, mode):
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
        rnn_helper.add_sequence_outputs(self.rnn_model, self.maximum_iterations, mode)

        # predictions, backpointers:
        self.rnn_model.build_tensor_array_loop_var("zero_tensor:|int", name="beamsearch_predictions", maximum_iterations=self.maximum_iterations)
        self.rnn_model.build_tensor_array_loop_var("zero_tensor:|int", name="beamsearch_backpointers", maximum_iterations=self.maximum_iterations)
        self.rnn_model.build_loop_var("zero_tensor:|float", name="beamsearch_propagated_scores")

        self.rnn_model.add_length_var()
        self.rnn_model.add_finished_var()
        self.rnn_model.add_counter_loop_var()

        n_rec = self.rnn_model.count_recurrent_links()
        n_out = self.rnn_model.count_output_links()

        loop = tf.while_loop(
            self.cond,
            self.body,
            loop_vars=tuple(self.rnn_model.loop_vars),
            maximum_iterations=self.maximum_iterations
        )

        prediction_index = n_rec + n_out
        backpointer_index = n_rec + n_out + 1

        pred_stack = loop[prediction_index].stack()
        lengths = loop[-3]

        backpointers = tf.reshape(loop[backpointer_index].stack(), [-1, self.rnn_model.batch_size, self.beam_width])

        max_length = tf.shape(backpointers)[0]
        mask = tf.where(tf.equal(pred_stack, tf.constant(self.stop_symbol, dtype=tf.int32)), tf.zeros_like(pred_stack), tf.ones_like(pred_stack))
        mask = tf.reshape(mask, [-1, self.rnn_model.batch_size, self.beam_width])
        lookup = self.get_selected_states(backpointers, lengths, max_length, mask)

        lengths, lookup = self.remove_extra_beams(lengths, lookup)
        max_length = tf.reduce_max(lengths)
        lookup = lookup[:max_length]

        decoded_sequences = self.apply_decoding(lookup, max_length, pred_stack)
        aux_out = self.get_aux_output_dict(lookup, max_length, loop)

        return decoded_sequences, lengths, aux_out

    def get_aux_output_dict(self, lookup, max_length, loop):
        n_rec = self.rnn_model.count_recurrent_links()
        n_out = self.rnn_model.count_output_links()

        aux_out = []
        for i in range(n_rec, n_rec + n_out):
            v = loop[i].stack()
            v = self.apply_decoding(lookup, max_length, v)
            aux_out.append(v)

        return {self.rnn_model.out_links[i][0]: aux_out[i] for i in range(len(self.rnn_model.out_links))}

    def apply_decoding(self, lookup, max_length, pred_stack):
        pred_stack = pred_stack[:max_length]
        decoded_sequences = tf.gather_nd(pred_stack, lookup)

        decode_shape = tf.concat([[-1, self.rnn_model.batch_size * self.n_to_output], tf.shape(tf.squeeze(decoded_sequences))[2:]], axis=-1)
        transpose_shape = tf.concat([[1,0], tf.range(2, tf.shape(decode_shape)[0])], axis=-1)
        decoded_sequences = tf.transpose(
            tf.reshape(decoded_sequences, decode_shape), transpose_shape)
        return decoded_sequences

    def remove_extra_beams(self, lengths, lookup):
        if self.n_to_output < self.beam_width:
            lookup = lookup[:, :, :self.n_to_output]
            lengths = tf.reshape(lengths, [-1, self.beam_width])
            lengths = lengths[:, :self.n_to_output]
            lengths = tf.reshape(lengths, [-1])
        return lengths, lookup

    def get_selected_states(self, backpointers, lengths, max_length, mask):
        #TODO: Decode end tokens properly
        beam_range = tf.range(self.beam_width, dtype=tf.int32)
        beam_states = tf.reshape(tf.tile(beam_range, [self.rnn_model.batch_size * max_length]),
                                 [-1, self.rnn_model.batch_size, self.beam_width])
        batch_range = tf.expand_dims(tf.expand_dims(tf.range(self.rnn_model.batch_size, dtype=tf.int32), -1), 0)
        tiled_batch_range = tf.tile(batch_range, [max_length, 1, self.beam_width])
        time_range = tf.expand_dims(tf.expand_dims(tf.range(max_length, dtype=tf.int32), -1), -1)
        tiled_time_range = tf.tile(time_range, [1, self.rnn_model.batch_size, self.beam_width])

        old_beam_states = beam_states

        beam_states = (beam_states + 1) * mask
        selected_beam_states = self.decode(beam_states, backpointers, tf.reshape(lengths, [-1, self.beam_width]), 0)

        selected_beam_states -= 1
        selected_beam_states = tf.where(tf.equal(selected_beam_states, tf.constant(-1, dtype=tf.int32)), old_beam_states, selected_beam_states)

        batch_beams = tiled_batch_range * self.beam_width + selected_beam_states
        batch_beams = batch_beams
        # Last element of every beam should be tiled_batch_range * self.beam_width + [0, 1, 2]
        lookup = tf.stack([tiled_time_range, batch_beams], -1)
        return lookup

    def decode(self, predictions, backpointers, lengths, stop_symbol):
        #TODO: Replace with own
        decoded_seqs = tf.contrib.seq2seq.gather_tree(predictions,
                                                      backpointers,
                                                      tf.reduce_max(lengths, axis=-1),
                                                      stop_symbol)

        return decoded_seqs

    def count_recurrences(self):
        return len(self.recurrences)

    def run_graph(self):
        results = self.graph.execute()
        return {output[0]: result for output, result in zip(self.out_links, results)}

    def compute_types(self, mode):
        inner_graph_output = self.rnn_model.get_inner_graph_output_types(mode)

        return inner_graph_output

    def set_graph_name(self, name):
        self.graph_name = name

    def get_populate_items(self):
        return [("graph", {"name": self.rnn_model.get_graph_name()})]

    def get_required_graph_outputs(self):
        return self.rnn_model.get_required_graph_outputs()

    def get_graph_inputs(self):
        return self.rnn_model.get_required_graph_inputs()