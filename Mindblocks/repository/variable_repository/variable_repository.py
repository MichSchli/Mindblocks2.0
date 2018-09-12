import operator
from functools import reduce

from Mindblocks.error_handling.repository.variable_not_found_exception import VariableNotFoundException
from Mindblocks.model.variable.variable_model import VariableModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository
from Mindblocks.repository.variable_repository.variable_specifications import VariableSpecifications


class VariableRepository(AbstractRepository):

    def __initialize_model__(self):
        return VariableModel()

    def get_specifications(self):
        return VariableSpecifications()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        return model

    def apply_search_configuration(self, search_configuration):
        self.logger_manager.log("Applying configuration:", "search", "status")
        for variable_name, search_field, search_option in search_configuration.iterate_options():
            variable = self.get_by_name(variable_name)[0]
            variable.set_search_option(search_field, search_option)
            updated_value = variable.get_value(mode="train")

        for variable in self.get_all():
            if variable.get_name() in search_configuration.get_affected_variables():
                self.logger_manager.log(" * " + variable.get_name() + " = \"" + updated_value + "\"", "search",
                                        "status")

    def describe_search_configuration(self, search_configuration):
        # TODO: Just applying is wrongm should reset to what we had before
        self.logger_manager.log("Applying configuration:", "search", "status")
        for variable_name, search_field, search_option in search_configuration.iterate_options():
            variable = self.get_by_name(variable_name)[0]
            variable.set_search_option(search_field, search_option)
            updated_value = variable.get_value(mode="train")

        for variable in self.get_all():
            if variable.get_name() in search_configuration.get_affected_variables():
                self.logger_manager.log(" * " + variable.get_name() + " = \"" + updated_value + "\"", "search", "status")

    def count_search_options(self, mode=None, greedy=True):
        option_list = []

        for variable in self.get_all():
            for option_count in variable.count_search_options(mode=mode):
                option_list.append(option_count)

        if greedy:
            return sum(option_list) - len(option_list) + 1
        else:
            return reduce(operator.mul, option_list, 1)

    def set_variable_value(self, name, value, mode=None):
        variables = self.get_by_name(name)

        if len(variables) == 0:
            raise VariableNotFoundException("Attempted to edit undeclared variable \"" + name + "\".")

        variable = variables[0]
        variable.set_value(value, mode=mode)