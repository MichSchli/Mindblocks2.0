import os

import sys


class Logger:

    filename = None

    def __init__(self, configuration, filename):
        self.configuration = configuration
        self.filename = filename

        if self.filename is not None:
            save_dir = os.path.dirname(self.filename)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            if os.path.isfile(self.filename):
                os.remove(self.filename)

    def log(self, message, context, subcontext):
        if context in self.configuration:
            if subcontext in self.configuration[context] or "all" in self.configuration[context]:
                if self.filename is None:
                    print(message, flush=True)
                else:
                    save_dir = os.path.dirname(self.filename)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    with open(self.filename, 'a') as log_file:
                        print(message, file=log_file)