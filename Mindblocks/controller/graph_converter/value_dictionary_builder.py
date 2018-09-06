from Mindblocks.repository.graph_repository.graph_specifications import GraphSpecifications


class ValueDictionaryBuilder:

    variable_repository = None

    def __init__(self, variable_repository, graph_repository, logger_manager):
        self.variable_repository = variable_repository
        self.graph_repository = graph_repository
        self.logger_manager = logger_manager

    def initialize_values(self, execution_graphs):
        value_dictionary = {}

        for execution_graph in execution_graphs:
            for execution_object in execution_graph.get_execution_objects():
                creation_value = execution_object.get_value_dictionary()
                mode = execution_object.get_mode()

                if self.requires_unique_value(creation_value, mode):
                    build_mode = mode
                else:
                    build_mode = "default"

                value_ref = str(execution_object.get_origin_identifier()) + build_mode

                if value_ref not in value_dictionary:
                    value_model = self.build_value_model(execution_object, build_mode)
                    value_dictionary[value_ref] = value_model
                else:
                    value_model = value_dictionary[value_ref]

                execution_object.set_value_model(value_model)

    def requires_unique_value(self, creation_value, mode):
        for k, v in creation_value.items():
            for variable in self.get_all_variables():
                if self.referenced_in(v, variable):
                    if variable.unique_for(mode):
                        return True

        return False

    def referenced_in(self, v, variable):
        ref = False
        for val in v:
            if variable.referenced_in(val[0]):
                ref = True
            for key, attribute in val[1].items():
                if variable.referenced_in(attribute):
                    ref = True
                    break
            if ref:
                break
        return ref

    def get_all_variables(self):
        return self.variable_repository.get_all()

    def build_value_model(self, execution_object, mode):
        self.logger_manager.log("Building value model for " + execution_object.get_description(), "graph_construction", "value")
        updated_dict = {}

        for k,v in execution_object.get_value_dictionary().items():
            updated_list = v
            for variable in self.get_all_variables():
                index = 0
                for item, attributes in updated_list:
                    replaced_item = variable.replace_in_string(item, mode=mode)
                    replaced_attributes = {idx: variable.replace_in_string(attr, mode=mode) for idx, attr in attributes.items()}
                    updated_list[index] = (replaced_item, replaced_attributes)
                    index += 1

            updated_dict[k] = updated_list

        value = execution_object.initialize_value(updated_dict, mode)

        return value