import os


class FilepathHandler:

    def get_test_block_path(self, filename):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path = dirname.split("/")[:-2]

        path.append("test_blocks")
        path.append(filename)

        return "/".join(path)