from collections import OrderedDict
import torch


def extract_submodel_state(global_state, local_state):
    """
    Slice the full global model into the smaller local model shape.
    """
    sub_state = OrderedDict()

    for name, local_param in local_state.items():
        global_param = global_state[name].detach().cpu()

        if global_param.ndim == 4:
            sub_state[name] = global_param[
                : local_param.shape[0],
                : local_param.shape[1],
                :, :
            ].clone()

        elif global_param.ndim == 2:
            sub_state[name] = global_param[
                : local_param.shape[0],
                : local_param.shape[1]
            ].clone()

        elif global_param.ndim == 1:
            sub_state[name] = global_param[: local_param.shape[0]].clone()

        elif global_param.ndim == 0:
            sub_state[name] = global_param.clone()

        else:
            sub_state[name] = global_param.clone()

    return sub_state


def add_submodel_to_global(accum_state, count_state, client_state):
    """
    Add client submodel weights into matching slices of the full global model.
    """
    for name, client_param in client_state.items():
        client_param = client_param.detach().cpu()

        # Integer buffers like BatchNorm num_batches_tracked are copied, not averaged.
        if not torch.is_floating_point(client_param):
            if client_param.ndim == 0:
                accum_state[name] = client_param.clone()
                count_state[name] += 1
            elif client_param.ndim == 1:
                accum_state[name][: client_param.shape[0]] = client_param
                count_state[name][: client_param.shape[0]] += 1
            else:
                accum_state[name] = client_param.clone()
                count_state[name] += 1
            continue

        if client_param.ndim == 4:
            accum_state[name][
                : client_param.shape[0],
                : client_param.shape[1],
                :, :
            ] += client_param

            count_state[name][
                : client_param.shape[0],
                : client_param.shape[1],
                :, :
            ] += 1

        elif client_param.ndim == 2:
            accum_state[name][
                : client_param.shape[0],
                : client_param.shape[1]
            ] += client_param

            count_state[name][
                : client_param.shape[0],
                : client_param.shape[1]
            ] += 1

        elif client_param.ndim == 1:
            accum_state[name][: client_param.shape[0]] += client_param
            count_state[name][: client_param.shape[0]] += 1

        elif client_param.ndim == 0:
            accum_state[name] += client_param
            count_state[name] += 1

    return accum_state, count_state


def average_submodels(global_state, client_states):
    """
    Slice-aware HeteroFL aggregation.
    Floating tensors are averaged. Integer buffers are copied from an updated client
    when available, otherwise kept from the previous global state.
    """
    accum_state = OrderedDict()
    count_state = OrderedDict()

    for name, param in global_state.items():
        param = param.detach().cpu()
        accum_state[name] = torch.zeros_like(param)
        count_state[name] = torch.zeros_like(param, dtype=torch.float32)

    for client_state in client_states:
        accum_state, count_state = add_submodel_to_global(
            accum_state,
            count_state,
            client_state,
        )

    new_state = OrderedDict()

    for name, old_param in global_state.items():
        old_param = old_param.detach().cpu()
        count = count_state[name]
        mask = count > 0

        new_param = old_param.clone()

        if torch.is_floating_point(old_param):
            new_param[mask] = accum_state[name][mask] / count[mask].to(accum_state[name].dtype)
        else:
            new_param[mask] = accum_state[name][mask].to(new_param.dtype)

        new_state[name] = new_param

    return new_state
