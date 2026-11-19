"""Model registry and factory functions."""

from typing import Callable, Dict
from torch import nn

from fl_bench.models.femnist_cnn import FEMNISTCNN, create_femnist_cnn
from fl_bench.models.resnet18 import create_resnet18
from fl_bench.models.shakespeare_lstm import (
    ALL_LETTERS,
    NUM_LETTERS,
    ShakespeareLSTM,
    create_shakespeare_lstm,
)

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": create_resnet18,
    "femnist_cnn": create_femnist_cnn,
    "shakespeare_lstm": create_shakespeare_lstm,
}


def register_model(name: str):
    """Decorator to register a new model constructor."""

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def create_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Supported: {supported}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_num_bytes(model: nn.Module) -> int:
    """Return total byte size of all tensors in `model.state_dict()`."""
    return sum(
        int(t.numel()) * int(t.element_size())
        for t in model.state_dict().values()
    )


def trainable_num_bytes(model: nn.Module) -> int:
    """Return total byte size of all trainable parameter tensors (`requires_grad=True`)."""
    return sum(
        int(p.numel()) * int(p.element_size())
        for p in model.parameters()
        if p.requires_grad
    )


__all__ = [
    "ALL_LETTERS",
    "FEMNISTCNN",
    "MODEL_REGISTRY",
    "NUM_LETTERS",
    "ShakespeareLSTM",
    "count_parameters",
    "create_femnist_cnn",
    "create_model",
    "create_resnet18",
    "create_shakespeare_lstm",
    "register_model",
    "state_dict_num_bytes",
    "trainable_num_bytes",
]
