import random

from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class ListNegativeSampler(ComponentTypeModel):

    name = "ListNegativeSampler"
    in_sockets = ["list"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        value = ListNegativeSamplerValue()

        if "is_gold_column" in value_dictionary:
            value.set_gold_column(int(value_dictionary["is_gold_column"][0][0]))

        if "pos_sample_rate" in value_dictionary:
            value.set_pos_sample_rate(int(value_dictionary["pos_sample_rate"][0][0]))

        if "neg_sample_rate" in value_dictionary:
            value.set_neg_sample_rate(int(value_dictionary["neg_sample_rate"][0][0]))

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        input_list = input_dictionary["list"]
        input_values = input_list.get_value()

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
        o = input_types["list"].copy()

        sample_size = value.pos_sample_rate + value.neg_sample_rate

        return {"output": o}


class ListNegativeSamplerValue(ExecutionComponentValueModel):

    def __init__(self):
        self.gold_column = -1
        self.pos_sample_rate = 1
        self.neg_sample_rate = 1

    def set_gold_column(self, idx):
        self.gold_column = idx

    def set_neg_sample_rate(self, rate):
        self.neg_sample_rate = rate

    def set_pos_sample_rate(self, rate):
        self.neg_sample_rate = rate