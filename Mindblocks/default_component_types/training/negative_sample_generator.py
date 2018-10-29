import random

from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel
import numpy as np


class ListNegativeSampler(ComponentTypeModel):

    name = "NegativeSampleGenerator"
    in_sockets = ["tensor"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = ListNegativeSamplerValue()

        if "gold_column" in value_dictionary:
            value.set_gold_column(int(value_dictionary["gold_column"][0][0]))

        if "positive_sample_rate" in value_dictionary:
            value.set_pos_sample_rate(int(value_dictionary["positive_sample_rate"][0][0]))

        if "negative_sample_rate" in value_dictionary:
            value.set_neg_sample_rate(int(value_dictionary["negative_sample_rate"][0][0]))

        return value

    def recursive_sample(self, elements, length_tensor_list, current_prefix, sample_count):
        local_elements = elements
        for dim in current_prefix:
            local_elements = local_elements[dim]

        if len(local_elements.shape) == 1:
            positive_indexes = np.nonzero(local_elements)[0]
            max_samples = min(sample_count, positive_indexes.shape[0])

            if max_samples > 0:
                positive_idx_sample = positive_indexes[np.random.choice(positive_indexes.shape[0], size=max_samples, replace=False)]
            else:
                positive_idx_sample = np.array([], dtype=np.int32)
            return positive_idx_sample
        else:
            local_length = length_tensor_list[len(current_prefix)]
            if local_length is not None:
                for idx in current_prefix:
                    local_length = local_length[idx]
            else:
                local_length = local_elements.shape[0]

                l = []
                for inner_elem_idx in range(local_length):
                    inner_sample = self.recursive_sample(elements, length_tensor_list, current_prefix + (inner_elem_idx,), sample_count)
                    l.append(inner_sample)

                return l

    def recursive_combine(self, left, right, length_tensor_list, current_prefix, depth):
        local_left = left
        local_right = right
        for dim in current_prefix:
            local_left = local_left[dim]
            local_right = local_right[dim]

        if len(current_prefix) == depth:
            return np.concatenate((local_left, local_right))
        else:
            local_length = length_tensor_list[len(current_prefix)]
            if local_length is not None:
                for idx in current_prefix:
                    local_length = local_length[idx]
            else:
                local_length = len(local_left)

                l = []
                for inner_elem_idx in range(local_length):
                    inner_sample = self.recursive_combine(left, right, length_tensor_list, current_prefix + (inner_elem_idx,), depth)
                    l.append(inner_sample)

                return l

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        input_list = input_dictionary["tensor"]
        input_values = input_list.get_value()
        lengths = input_list.get_lengths()

        pos_sample = self.recursive_sample(input_values, lengths, (), value.pos_sample_rate)

        sth = SoftTensorHelper()
        replacement_tensor = np.ones_like(input_values)
        replaced = sth.python_replace_elements_outside_lengths(input_values, lengths, replacement_tensor)
        print(np.logical_not(replaced))
        neg_sample = self.recursive_sample(np.logical_not(replaced), lengths, (), value.neg_sample_rate)

        print(input_values)
        print(pos_sample)

        print(neg_sample)

        combined_samples = self.recursive_combine(pos_sample, neg_sample, lengths, (), len(input_values.shape)-1)
        print(combined_samples)
        exit()

        sth = SoftTensorHelper()
        replacement_tensor = np.ones_like(input_values)
        replaced = sth.python_replace_elements_outside_lengths(input_values, lengths, replacement_tensor)
        negative_indexes = np.nonzero(np.logical_not(replaced))

        print(input_values)
        #print(negative_indexes)
        exit()

        new_out = []

        for batch in input_values:
            positive_indexes = [i for i in range(len(batch)) if batch[i][value.gold_column] == "True"]
            negative_indexes = [i for i in range(len(batch)) if batch[i][value.gold_column] != "True"]

            random.shuffle(negative_indexes)
            random.shuffle(positive_indexes)

            negative_indexes = negative_indexes[:value.sample_rate]

            if not value.use_all_golds:
                positive_indexes = positive_indexes[:1]

            new_negs = [batch[j] for j in negative_indexes]
            new_pos = [batch[j] for j in positive_indexes]

            new_out.append(new_pos + new_negs)

        output_models["output"].assign(new_out)
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        input_dims = input_types["tensor"].get_dimensions()
        input_soft = input_types["tensor"].get_soft_by_dimensions()

        sample_size = value.pos_sample_rate + value.neg_sample_rate

        output_dims = input_dims[:]
        output_soft = input_soft[:]

        if value.gold_column is not None:
            output_dims = output_dims[:-1]
            output_soft = output_soft[:-1]

        output_dims[-1] = sample_size
        output_soft[-1] = True

        output_tensor = SoftTensorTypeModel(output_dims,
                                            string_type="int",
                                            soft_by_dimensions=output_soft)

        return {"output": output_tensor}


class ListNegativeSamplerValue(ExecutionComponentValueModel):

    gold_column = None
    pos_sample_rate = 1
    neg_sample_rate = 1

    def __init__(self):
        self.gold_column = None
        self.pos_sample_rate = 1
        self.neg_sample_rate = 1

    def set_gold_column(self, idx):
        self.gold_column = idx

    def set_neg_sample_rate(self, rate):
        self.neg_sample_rate = rate

    def set_pos_sample_rate(self, rate):
        self.neg_sample_rate = rate