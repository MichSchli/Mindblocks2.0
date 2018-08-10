from tensorflow.python.ops import tensor_array_ops

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.tensor.tensor_type_model import TensorTypeModel
import tensorflow as tf

class ScheduledSamplingRnnComponent(ComponentTypeModel):

    name = "ScheduledSamplingRnn"
    in_sockets = ["teacher_inputs"]
    out_sockets = []
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = ScheduledSamplingRnnComponentValue()
        value.set_graph_name(value_dictionary["graph"][0][0])
        for in_link in value_dictionary["in_link"]:
            parts = in_link[0].split("->")
            feed_type = in_link[1]["feed"] if "feed" in in_link[1] else None
            value.add_in_link(parts[0], parts[1], feed_type=feed_type)

        for out_link in value_dictionary["out_link"]:
            parts = out_link[0].split("->")
            feed_type = out_link[1]["feed"] if "feed" in out_link[1] else None
            value.add_out_link(parts[1], parts[0], feed_type=feed_type)

        for recurrence in value_dictionary["recurrence"]:
            parts = recurrence[0].split("->")
            init = recurrence[1]["init"] if "init" in recurrence[1] else None
            value.add_recurrence(parts[0], parts[1], init=init)

        if "batch_size" in value_dictionary:
            value.batch_size = int(value_dictionary["batch_size"][0][0])

        return value

    def execute(self, input_dictionary, value, output_models, mode):
        outputs = value.assign_and_run(input_dictionary)

        for k,v in outputs.items():
            output_models[k].assign(v, language="tensorflow")

        print(output_models)

        return output_models

    def build_value_type_model(self, input_types, value):
        value.assign_input_types(input_types)
        output_types = value.compute_types()

        return output_types


class ScheduledSamplingRnnComponentValue:

    graph_name = None
    graph = None
    batch_size = None

    def __init__(self):
        self.in_links = []
        self.out_links = []
        self.recurrences = []

    def add_in_link(self, component_input, graph_input, feed_type=None):
        self.in_links.append((component_input, graph_input, feed_type))

    def add_out_link(self, component_output, graph_output, feed_type=None):
        self.out_links.append((component_output, graph_output, feed_type))

    def add_recurrence(self, graph_output, graph_input, init):
        self.recurrences.append((graph_output, graph_input, init))

    def assign_input_types(self, input_dictionary):
        batch_size = input_dictionary["teacher_inputs"].get_batch_size()
        max_sequence_length = input_dictionary["teacher_inputs"].get_maximum_sequence_length()

        for component_input, graph_input, feed_type in self.in_links:
            parts = graph_input.split(":")
            source_input_type = input_dictionary[component_input]

            if feed_type == "loop":
                graph_input_type = source_input_type.get_single_token_type()
            elif feed_type == "per_batch" or feed_type == "initializer":
                graph_input_type = source_input_type
            else:
                graph_input_type = source_input_type

            self.graph.enforce_type(parts[0], parts[1], graph_input_type)

        for graph_output, graph_input, init in self.recurrences:
            if init is not None and init.startswith("socket:"):
                parts = graph_input.split(":")
                self.graph.enforce_type(parts[0], parts[1], graph_input_type)

        for graph_output, graph_input, init in self.recurrences:
            if init is not None and init.startswith("zero_tensor"):
                parts = graph_input.split(":")
                init_info = init[12:].split("|")
                init_type = init_info[1] if len(init_info) > 1 else "float"

                dims = [batch_size] + [int(v) for v in init_info[0].split(",")] if len(init_info[0]) > 0 else [batch_size]
                tensor_type = TensorTypeModel(init_type, dims)
                self.graph.enforce_type(parts[0], parts[1], tensor_type)

    def set_nths_input(self, n, value):
        if n >= len(self.list_of_in_sockets):
            return

        in_socket = self.list_of_in_sockets[n]
        type = in_socket.replaced_type
        value_model = type.initialize_value_model()
        value_model.assign(value)
        in_socket.replace_value(value_model)

    def body(self, *args):
        print(args)
        for n,arg in enumerate(args):
            if n > len(self.out_links):
                self.set_nths_input(n - len(self.out_links) - 1, arg)

        results = self.graph.execute(discard_value_models=True)

        for i in range(len(results)):
            if i < len(self.out_links):
                results[i] = self.write_to_tensor_array(args[i+1],results[i], args[0])

        return (args[0] + 1, ) + tuple(results)

    def write_to_tensor_array(self, array, item, index):
        return array.write(index, item)

    def cond(self, *args):
        print("cond")
        return True

    def assign_and_run(self, input_dictionary):
        batch_size = input_dictionary["teacher_inputs"].get_batch_size()
        maximum_iterations = input_dictionary["teacher_inputs"].get_maximum_sequence_length()

        sequence_feeds = []
        sequence_sockets = []

        loop_var_initializers = []
        self.list_of_in_sockets = []

        for component_input, graph_input, feed_type in self.in_links:
            parts = graph_input.split(":")
            if feed_type == "loop":
                sequence_feeds.append(input_dictionary[component_input])
                sequence_sockets.append((parts[0], parts[1]))
            elif feed_type == "initializer":
                loop_var_initializers.append((parts[0], parts[1], input_dictionary[component_input].get_value()))
                self.list_of_in_sockets.append(self.graph.get_in_socket(parts[0], parts[1]))
            else:
                self.graph.enforce_value(parts[0], parts[1], input_dictionary[component_input])

        for graph_output, graph_input, init in self.recurrences:
            if init is not None and init.startswith("zero_tensor"):
                parts = graph_input.split(":")
                in_socket = self.graph.get_in_socket(parts[0], parts[1])
                dims = in_socket.replaced_type.get_dimensions()
                tf_type = tf.int32 if in_socket.replaced_type.type == "int" else tf.float32
                tf_value = tf.zeros(dims, dtype=tf_type)
                loop_var_initializers.append((parts[0], parts[1], tf_value))
                self.list_of_in_sockets.append(in_socket)
            elif init is not None and init.startswith("socket"):
                parts = graph_input.split(":")
                in_socket = self.graph.get_in_socket(parts[0], parts[1])
                linked_socket = input_dictionary[init[7:]]

                loop_var_initializers.append((parts[0], parts[1], linked_socket.get_value()))
                self.list_of_in_sockets.append(in_socket)

        loop_vars = tuple(x[2] for x in loop_var_initializers)

        for _, graph_output, _ in self.out_links:
            # use tensor arrays
            parts = graph_output.split(":")
            socket = self.graph.get_out_socket(parts[0], parts[1])
            out_type = socket.pull_type_model()
            dims = out_type.get_dimensions()
            tf_value = tensor_array_ops.TensorArray(
                dtype=tf.float32,
                size=0 if maximum_iterations is None else maximum_iterations,
                dynamic_size=maximum_iterations is None,
                element_shape=dims)
            loop_vars = (tf_value, ) + loop_vars

        print(sequence_sockets)
        print(sequence_feeds)
        print(loop_var_initializers)

        loop_vars = (0,) + loop_vars

        loop = tf.while_loop(
            self.cond,
            self.body,
            loop_vars=loop_vars,
            maximum_iterations=maximum_iterations
        )

        loop = list(loop)[1:]

        for i in range(len(loop)):
            if i < len(self.out_links):
                loop[i] = tf.reshape(loop[i].stack(), [batch_size, maximum_iterations, -1])

        return {self.out_links[i][0]: loop[i] for i in range(len(self.out_links))}

    def count_recurrences(self):
        return len(self.recurrences)

    def run_graph(self):
        results = self.graph.execute()
        return {output[0]: result for output, result in zip(self.out_links, results)}

    def compute_types(self):
        results = self.graph.initialize_type_models()
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