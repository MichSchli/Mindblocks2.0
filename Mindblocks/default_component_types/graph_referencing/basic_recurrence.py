from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class BasicRecurrenceComponent(ComponentTypeModel):

    name = "BasicRecurrence"
    in_sockets = []
    out_sockets = []
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = BasicRecurrenceComponentValue()
        value.set_graph_name(value_dictionary["graph"][0][0])
        for in_link in value_dictionary["in_link"]:
            parts = in_link[0].split("->")
            feed_type = in_link[1]["feed"] if "feed" in in_link[1] else None
            value.add_in_link(parts[0], parts[1], feed_type=feed_type)

        for out_link in value_dictionary["out_link"]:
            parts = out_link[0].split("->")
            feed_type = in_link[1]["feed"] if "feed" in in_link[1] else None
            value.add_out_link(parts[1], parts[0], feed_type=feed_type)

        for recurrence in value_dictionary["recurrence"]:
            parts = recurrence[0].split("->")
            value.add_recurrence(parts[0], parts[1])

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        outputs = value.assign_and_run(input_dictionary)

        for k,v in outputs.items():
            output_models[k].assign(v)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        value.assign_input_types(input_types)
        output_types = value.compute_types(mode)

        return output_types

    def get_used_in_sockets(self, value):
        return []


class BasicRecurrenceComponentValue(ExecutionComponentValueModel):

    graph_name = None
    graph = None

    def set_graph(self, graph):
        self.graph = graph

    def __init__(self):
        self.in_links = []
        self.out_links = []
        self.recurrences = []

    def get_referenced_graphs(self):
        return [self.graph]

    def add_in_link(self, component_input, graph_input, feed_type=None):
        self.in_links.append((component_input, graph_input, feed_type))

    def add_out_link(self, component_output, graph_output, feed_type=None):
        self.out_links.append((component_output, graph_output, feed_type))

    def add_recurrence(self, graph_output, graph_input):
        self.recurrences.append((graph_output, graph_input))

    def assign_input_types(self, input_dictionary):
        for component_input, graph_input, feed_type in self.in_links:
            parts = graph_input.split(":")
            source_input_type = input_dictionary[component_input]

            if feed_type == "loop":
                graph_input_type = source_input_type.get_single_token_type()
            else:
                graph_input_type = source_input_type

            self.graph.enforce_type(parts[0], parts[1], graph_input_type)

    def assign_and_run(self, input_dictionary):
        # INITIALIZE
        sequence_feeds = []
        sequence_sockets = []
        required_output_length = len(self.out_links)

        for component_input, graph_input, feed_type in self.in_links:
            parts = graph_input.split(":")
            if feed_type == "loop":
                sequence_feeds.append(input_dictionary[component_input])
                sequence_sockets.append((parts[0], parts[1]))
            else:
                self.graph.enforce_value(parts[0], parts[1], input_dictionary[component_input])

        # LOOP
        output_sequences = [[] for _ in range(required_output_length)]
        for i in range(sequence_feeds[0].get_batch_size()):
            seq_len = sequence_feeds[0].get_sequence_lengths()[i]

            for o in output_sequences:
                o.append([None]*seq_len)

            for component_input, graph_input, feed_type in self.in_links:
                parts = graph_input.split(":")
                if not feed_type == "loop":
                    self.graph.enforce_value(parts[0], parts[1], input_dictionary[component_input])

            for token_index in range(seq_len):
                for feed, socket_dec in zip(sequence_feeds, sequence_sockets):
                    current_batch_feed = feed.get_token(i, token_index)
                    self.graph.enforce_value(socket_dec[0], socket_dec[1], current_batch_feed)

                results = self.graph.execute(discard_value_models=False)
                to_output = results[:self.count_recurrences()]

                recurring_outputs = results[self.count_recurrences():]
                for rec, output in zip(self.recurrences, recurring_outputs):
                    parts = rec[1].split(":")
                    self.graph.enforce_value(parts[0], parts[1], output)

                for j in range(required_output_length):
                    output_sequences[j][i][token_index] = to_output[j].get_value()

        out_dict = {}
        for i in range(required_output_length):
            out_dict[self.out_links[i][0]] = output_sequences[i]
        return out_dict

    def count_recurrences(self):
        return len(self.recurrences)

    def run_graph(self):
        results = self.graph.execute()
        return {output[0]: result for output, result in zip(self.out_links, results)}

    def compute_types(self, mode):
        results = self.graph.initialize_type_models(mode)
        out_type_dict = {}
        for output, result in zip(self.out_links, results):
            component_output, _, feed_type = output

            if feed_type == "loop":
                out_type = result.to_sequence_type()
            else:
                out_type = result

            out_type_dict[component_output] = out_type

        return out_type_dict

    def set_graph_name(self, name):
        self.graph_name = name

    def get_populate_items(self):
        return [("graph", {"name": self.graph_name})]

    def get_required_graph_outputs(self):
        return [(l[1].split(":")[0], l[1].split(":")[1]) for l in self.out_links] + \
               [(l[0].split(":")[0], l[0].split(":")[1]) for l in self.recurrences]

    def get_graph_inputs(self):
        return [(l[1].split(":")[0], l[1].split(":")[1]) for l in self.in_links]