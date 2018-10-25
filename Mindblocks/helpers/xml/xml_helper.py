class XmlHelper:

    ESCAPE = {"&quot;": "\"",
              "&apos;": "\'",
              "&lt;": "<",
              "&gt;": ">",
              "&amp;": "&",
              "\\n": "\n",
              "\\t": "\t"}

    def get_header(self, name, fields={}, indentation=0):
        string = "<"+name

        if fields != {}:
            string += " " + " ".join([str(k)+"="+str(v) for k,v in fields.items()])

        string += ">"

        return self.add_indentation(string, indentation)

    def get_footer(self, name, indentation=0):
        return self.add_indentation("</"+name+">", indentation)

    def add_indentation(self, string, indentation):
        indentation = '\t' * indentation
        return indentation + string

    def process_text(self, text):
        for k,v in self.ESCAPE.items():
            text = text.replace(k, v)

        return text

    def read_symbol(self, lines, start_index=0):
        if start_index >= len(lines):
            return None

        scanner = start_index
        is_symbol = lines[start_index] == "<"

        if is_symbol:
            while lines[scanner] != ">":
                scanner += 1
            scanner += 1
            symbol = lines[start_index:scanner]
            parts = symbol[1:-1].split(' ')
            name = parts[0]

            name = self.process_text(name)
            return name
        else:
            while lines[scanner] != "<":
                scanner += 1
            symbol = lines[start_index:scanner]
            return symbol

    def pop_symbol(self, lines, start_index=0):
        scanner = start_index
        is_symbol = lines[start_index] == "<"

        if is_symbol:
            while lines[scanner] != ">":
                scanner += 1
            scanner += 1
            symbol = lines[start_index:scanner]
            parts = symbol[1:-1].split(' ')
            name = parts[0]
            attributes = [tuple(att.split("=")) for att in parts[1:]]
            attributes = {k: self.process_text(v.replace("\"", "")) for k,v in attributes}

            name = self.process_text(name)

            return name, attributes, scanner
        else:
            while lines[scanner] != "<":
                scanner += 1
            symbol = lines[start_index:scanner]
            return self.process_text(symbol), [], scanner