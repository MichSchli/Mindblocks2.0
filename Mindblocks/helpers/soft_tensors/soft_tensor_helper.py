import numpy as np

class SoftTensorHelper:

    def recursive_transform(self, old_representation, new_representation, current_prefix, length_list, stop_dim, transform_fn):
        local_old_representation = old_representation
        for idx in current_prefix:
            local_old_representation = local_old_representation[idx]

        if len(current_prefix) == stop_dim or len(current_prefix) - len(old_representation.shape) -1 == stop_dim:
            new_representation[current_prefix] = transform_fn(local_old_representation)
        else:
            local_length = length_list[len(current_prefix)]
            if local_length is not None:
                for idx in current_prefix:
                    local_length = local_length[idx]
            else:
                local_length = local_old_representation.shape[0]

            for inner_elem_idx in range(local_length):
                self.recursive_transform(old_representation, new_representation, current_prefix + (inner_elem_idx, ), length_list, stop_dim, transform_fn)

    def transform(self, old_tensor, length_list, transform, new_type=np.float32, stop_dim=-1):

        if stop_dim == -1:
            new_dims = old_tensor.shape
        else:
            new_dims = old_tensor.shape[:stop_dim+1]
        new_tensor = np.zeros(new_dims, dtype=new_type)

        self.recursive_transform(old_tensor, new_tensor, (), length_list, stop_dim, transform)

        return new_tensor

    def recursive_python_list_build(self, old_representation, length_list, current_prefix):
        local_old_representation = old_representation

        for idx in current_prefix:
            local_old_representation = local_old_representation[idx]

        old_rep_shape = np.array(old_representation).shape
        if len(current_prefix) == len(old_rep_shape):
            return local_old_representation
        else:
            local_length = length_list[len(current_prefix)]
            if local_length is not None:
                for idx in current_prefix:
                    local_length = local_length[idx]
            else:
                local_length = np.array(local_old_representation).shape[0]

            curr_list = []
            for inner_elem_idx in range(local_length):
                curr_list.append(self.recursive_python_list_build(old_representation, length_list, current_prefix + (inner_elem_idx,)))

            return curr_list

    def format_to_python_list(self, tensor, length_list):
        rec_build = self.recursive_python_list_build(tensor, length_list, ())

        return rec_build