"""
HeteroFL top-left subnetwork index computation (`split_model`), subnetwork
extraction (`extract_subnetwork_state`), and parameter-aligned overlap aggregation
(`combine_subnetwork_states`) (Diao et al., ICLR 2021).
Reference: https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/blob/master/src/fed.py
"""

from collections import OrderedDict
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from fl_bench.methods.base import StateDict

ParamIndex = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def compute_subnetwork_indices(
    global_state: StateDict,
    model_name: str,
    scaler_rate: float,
) -> Dict[str, ParamIndex]:
    """
    Compute the top-left subnetwork parameter index mapping `param_idx[k]` for a single
    client with relative width rate `scaler_rate = r_m / r_g in (0, 1]`, matching
    `Federation.split_model()` in `src/fed.py:26-112`.
    """
    name = str(model_name).strip().lower()
    idx_map: Dict[str, ParamIndex] = OrderedDict()

    if name == "resnet18":
        curr_idx: Optional[torch.Tensor] = None
        for k, v in global_state.items():
            if not torch.is_floating_point(v) or "running_" in k or "num_batches_tracked" in k:
                continue
            param_type = k.split(".")[-1]
            if param_type == "weight":
                if v.dim() > 1:
                    in_size = int(v.size(1))
                    out_size = int(v.size(0))
                    if "conv1" in k or "conv2" in k:
                        if curr_idx is None:
                            curr_idx = torch.arange(in_size, device=v.device)
                        in_idx = curr_idx
                        local_out = max(1, int(math.ceil(out_size * scaler_rate)))
                        out_idx = torch.arange(out_size, device=v.device)[:local_out]
                        curr_idx = out_idx
                    elif "downsample.0" in k or "shortcut" in k:
                        conv1_key = k.replace("downsample.0", "conv1").replace("shortcut", "conv1")
                        conv1_pair = idx_map[conv1_key]
                        assert isinstance(conv1_pair, tuple)
                        in_idx = conv1_pair[1]
                        assert curr_idx is not None
                        out_idx = curr_idx
                    elif k.startswith("fc.") or "linear" in k:
                        assert curr_idx is not None
                        in_idx = curr_idx
                        out_idx = torch.arange(out_size, device=v.device)
                    else:
                        raise ValueError(f"Unexpected ResNet weight key '{k}'")
                    idx_map[k] = (out_idx, in_idx)
                else:
                    assert curr_idx is not None
                    idx_map[k] = curr_idx
            elif param_type == "bias":
                if k.startswith("fc.") or "linear" in k:
                    idx_map[k] = torch.arange(int(v.size(0)), device=v.device)
                else:
                    assert curr_idx is not None
                    idx_map[k] = curr_idx
        return idx_map

    if name == "femnist_cnn":
        # FEMNISTCNN: features.0 (Conv 1->32), features.3 (Conv 32->64),
        # classifier.1 (Linear 49*64 -> 2048), classifier.3 (Linear 2048 -> 62)
        c1_out = max(1, int(math.ceil(int(global_state["features.0.weight"].size(0)) * scaler_rate)))
        c2_out = max(1, int(math.ceil(int(global_state["features.3.weight"].size(0)) * scaler_rate)))
        h1_out = max(1, int(math.ceil(int(global_state["classifier.1.weight"].size(0)) * scaler_rate)))
        n_cls = int(global_state["classifier.3.weight"].size(0))
        dev = global_state["features.0.weight"].device

        idx_in_0 = torch.arange(int(global_state["features.0.weight"].size(1)), device=dev)
        idx_c1 = torch.arange(c1_out, device=dev)
        idx_c2 = torch.arange(c2_out, device=dev)
        # Flattened 7x7 spatial map per channel c in 0..c2_out-1 occupies 0..49*c2_out-1
        idx_flat = torch.arange(49 * c2_out, device=dev)
        idx_h1 = torch.arange(h1_out, device=dev)
        idx_cls = torch.arange(n_cls, device=dev)

        idx_map["features.0.weight"] = (idx_c1, idx_in_0)
        idx_map["features.0.bias"] = idx_c1
        idx_map["features.3.weight"] = (idx_c2, idx_c1)
        idx_map["features.3.bias"] = idx_c2
        idx_map["classifier.1.weight"] = (idx_h1, idx_flat)
        idx_map["classifier.1.bias"] = idx_h1
        idx_map["classifier.3.weight"] = (idx_cls, idx_h1)
        idx_map["classifier.3.bias"] = idx_cls
        return idx_map

    raise ValueError(f"Unsupported model '{model_name}' for HeteroFL subnetwork indexing.")
def extract_subnetwork_state(
    global_state: StateDict,
    param_idx: Dict[str, ParamIndex],
) -> Tuple[StateDict, int, int]:
    """
    Slice `global_state` according to `param_idx` (`Federation.distribute()` in `src/fed.py:143-153`).
    Returns `(local_parameters, subnetwork_num_params, subnetwork_num_bytes)`.
    """
    local_param: StateDict = OrderedDict()
    total_params = 0
    total_bytes = 0

    for k, idx_spec in param_idx.items():
        v = global_state[k]
        if isinstance(idx_spec, tuple):
            out_idx, in_idx = idx_spec
            grid = torch.meshgrid(out_idx, in_idx, indexing="ij")
            sliced = v[grid].clone()
        else:
            sliced = v[idx_spec].clone()
        local_param[k] = sliced
        total_params += int(sliced.numel())
        total_bytes += int(sliced.numel()) * int(sliced.element_size())

    return local_param, total_params, total_bytes


def combine_subnetwork_states(
    global_state: StateDict,
    local_states: Sequence[StateDict],
    param_indices: Sequence[Dict[str, ParamIndex]],
    client_label_splits: Sequence[Optional[torch.Tensor]],
    *,
    model_name: str,
    use_label_mask: bool = True,
) -> StateDict:
    """
    Parameter-aligned HeteroFL server aggregation (`Federation.combine()` in `src/fed.py:155-235`).

    Each parameter element `(i, j)` in `global_state[k]` is averaged over only the active
    clients whose subnetwork index `param_idx[m][k]` covers `(i, j)`. When `use_label_mask=True`,
    the final classifier layer (`fc.weight`/`fc.bias` or `classifier.3.weight`/`classifier.3.bias`)
    aggregates each client's parameters only into the class rows `label_split[m]` present on that client.
    Entries with `count == 0` retain their previous global value.
    """
    name = str(model_name).strip().lower()
    output_weight_key = "fc.weight" if name == "resnet18" else "classifier.3.weight"
    output_bias_key = "fc.bias" if name == "resnet18" else "classifier.3.bias"

    new_global: StateDict = {}
    for k, v in global_state.items():
        if k not in param_indices[0]:
            new_global[k] = v.clone()
            continue

        count = torch.zeros_like(v, dtype=torch.float64)
        tmp_v = torch.zeros_like(v, dtype=torch.float64)

        for m in range(len(local_states)):
            loc_tensor = local_states[m][k].to(torch.float64)
            idx_spec = param_indices[m][k]
            lbl_split = client_label_splits[m]

            if isinstance(idx_spec, tuple):
                out_idx, in_idx = idx_spec
                if use_label_mask and k == output_weight_key and lbl_split is not None and lbl_split.numel() > 0:
                    masked_out = out_idx[lbl_split]
                    grid = torch.meshgrid(masked_out, in_idx, indexing="ij")
                    tmp_v[grid] += loc_tensor[lbl_split]
                    count[grid] += 1.0
                else:
                    grid = torch.meshgrid(out_idx, in_idx, indexing="ij")
                    tmp_v[grid] += loc_tensor
                    count[grid] += 1.0
            else:
                if use_label_mask and k == output_bias_key and lbl_split is not None and lbl_split.numel() > 0:
                    masked_idx = idx_spec[lbl_split]
                    tmp_v[masked_idx] += loc_tensor[lbl_split]
                    count[masked_idx] += 1.0
                else:
                    tmp_v[idx_spec] += loc_tensor
                    count[idx_spec] += 1.0

        updated = v.to(torch.float64).clone()
        mask_pos = count > 0.0
        updated[mask_pos] = tmp_v[mask_pos] / count[mask_pos]
        new_global[k] = updated.to(v.dtype)

    return new_global
