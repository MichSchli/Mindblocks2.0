import re

from Mindblocks.model.abstract.abstract_model import AbstractModel


class VariableModel(AbstractModel):

    identifier = None
    name = None

    values = None

    search_sections = None
    search_ids = None

    def __init__(self):
        self.values = {"default": None}
        self.search_sections = {"default": None}
        self.search_ids = {"default": None}

    def set_value(self, value, mode=None):
        value = str(value)
        search_sections, search_ids = self.handle_search(value)

        if not mode:
            self.values["default"] = value
            self.search_sections["default"] = search_sections
            self.search_ids["default"] = search_ids
        else:
            self.values[mode] = value
            self.search_sections[mode] = search_sections
            self.search_ids[mode] = search_ids

    def get_name(self):
        return self.name

    def set_search_option(self, field, value):
        # TODO: Hardcoded to set for all modes:
        for mode in ["default", "train", "test", "validate"]:
            if mode in self.values and self.values[mode] is not None:
                self.search_ids[mode][field] = value

    def count_search_options(self, mode=None):
        if mode is not None and mode in self.values:
            return [len(option[1]) for option in self.search_sections[mode]]
        else:
            return [len(option[1]) for option in self.search_sections["default"]]

    def get_value(self, mode=None):
        if mode is not None and mode in self.values:
            string_value = self.values[mode]
            search_sections = self.search_sections[mode]
            search_ids = self.search_ids[mode]
        else:
            string_value = self.values["default"]
            search_sections = self.search_sections["default"]
            search_ids = self.search_ids["default"]

        if string_value is None:
            return None

        for search_section, search_id in zip(search_sections, search_ids):
            string_value = string_value.replace(search_section[0], search_section[1][search_id], 1)

        return string_value

    def referenced_in(self, string):
        ref = "$" + self.name
        return ref in string

    def defined_for(self):
        defined_for_modes = []
        for key in self.values:
            if key != "default":
                defined_for_modes.append(key)
        return defined_for_modes

    def unique_for(self, mode):
        return mode in self.values

    def replace_in_string(self, string, mode=None):
        replacement = self.get_value(mode=mode)
        if replacement is None:
            return string
        else:
            target_part = "$" + self.name
            return string.replace(target_part, str(replacement))

    def handle_search(self, string_value):
        string_search_sections = re.findall(r"\{([^}]+)\}", string_value)

        search_sections = [self.process_search_section(search_section) for search_section in string_search_sections]
        search_section_indexes = [0] * len(string_search_sections)

        return search_sections, search_section_indexes

    def process_search_section(self, search_section):
        search_items = search_section.split(";")
        search_items = [i.strip() for i in search_items]
        return "{" + search_section + "}", search_items
