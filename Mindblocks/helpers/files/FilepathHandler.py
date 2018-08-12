import os


class FilepathHandler:

    def get_test_data_path(self, filename):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path = dirname.split("/")[:-3]

        path.append("test_data")
        path.append(filename)

        return "/".join(path)

    def get_test_block_path(self, filename):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path = dirname.split("/")[:-3]

        path.append("test_blocks")
        path.append(filename)

        return "/".join(path)

    def get_default_component_type_file(self, filename):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path = dirname.split("/")[:-2]

        path.append("default_component_types")
        path.append(filename)

        return "/".join(path)

    def get_default_component_type_folder(self):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path = dirname.split("/")[:-2]

        path.append("default_component_types")

        return "/".join(path)