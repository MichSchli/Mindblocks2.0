class VariableModel:

    identifier = None
    name = None

    values = None

    def __init__(self):
        self.values = {"default": None}

    def set_value(self, value, mode=None):
        if not mode:
            self.values["default"] = value
        else:
            self.values[mode] = value

    def get_value(self, mode=None):
        if mode is None or mode not in self.values:
            return self.values["default"]
        else:
            return self.values[mode]

    def replace_in_string(self, string, mode=None):
        if mode in self.values:
            replacement = self.values[mode]
        elif self.values["default"] is not None:
            replacement = self.values["default"]
        else:
            return string

        target_part = "$" + self.name

        return string.replace(target_part, replacement)