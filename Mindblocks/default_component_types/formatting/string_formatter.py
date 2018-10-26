from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np

class StringFormatter(ComponentTypeModel):

    name = "StringFormatter"
    in_sockets = []
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return StringFormatterValue(value_dictionary["action"][0][0])

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        sth = SoftTensorHelper()
        first_val = list(input_dictionary.values())[0].get_value()
        first_lengths = list(input_dictionary.values())[0].get_lengths()

        initialize_fn = lambda x: value.action[:]
        result = sth.transform(first_val,
                               first_lengths,
                               initialize_fn,
                               new_type=np.object,
                               transform_dim=-1)



        for k, v in input_dictionary.items():
            second_tensor = v.get_value()

            transform_fn = lambda x, y: x.replace("[" + k + "]", y)
            result = sth.transform_combine(result,
                                           second_tensor,
                                           first_lengths,
                                           transform_fn,
                                           new_type=np.object,
                                           transform_dim=-1)

        output_models["output"].assign(result, length_list=first_lengths)
        return output_models

    def build_value_type_model(self, input_types, value, mode):
        ins = list(input_types.values())[0]
        return {"output": ins.copy()}


class StringFormatterValue(ExecutionComponentValueModel):

    def __init__(self, action):
        self.action = action