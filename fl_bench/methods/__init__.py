"""FL method registry and factory."""

from typing import Dict, Type

from fl_bench.config import ExperimentConfig
from fl_bench.methods.base import (
    BaseAlgorithm,
    BaseMethod,
    ClientResult,
    StateDict,
    fedavg_aggregate,
    tensor_dict_norm,
)
from fl_bench.methods.fedadam.method import FedAdamAlgorithm, FedAdamMethod
from fl_bench.methods.fedavg.method import FedAvgAlgorithm, FedAvgMethod
from fl_bench.methods.fedavgm.method import FedAvgMAlgorithm, FedAvgMMethod
from fl_bench.methods.fedgps.method import FedGPSAlgorithm, FedGPSMethod
from fl_bench.methods.fedlesam.method import FedLESAMAlgorithm, FedLESAMMethod
from fl_bench.methods.fednova.method import FedNovaAlgorithm, FedNovaMethod
from fl_bench.methods.fedprox.method import FedProxAlgorithm, FedProxMethod
from fl_bench.methods.fedspeed.method import FedSpeedAlgorithm, FedSpeedMethod
from fl_bench.methods.fedwmsam.method import FedWMSAMAlgorithm, FedWMSAMMethod
from fl_bench.methods.heterofl.method import HeteroFLAlgorithm, HeteroFLMethod
from fl_bench.methods.scaffold.method import (
    ScaffoldAlgorithm,
    ScaffoldMethod,
    zero_control,
)

METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {
    "fedavg": FedAvgMethod,
    "fedprox": FedProxMethod,
    "fedavgm": FedAvgMMethod,
    "fedadam": FedAdamMethod,
    "fednova": FedNovaMethod,
    "scaffold": ScaffoldMethod,
    "fedspeed": FedSpeedMethod,
    "fedlesam": FedLESAMMethod,
    "fedgps": FedGPSMethod,
    "fedwmsam": FedWMSAMMethod,
    "heterofl": HeteroFLMethod,
}

ALGORITHM_REGISTRY = METHOD_REGISTRY


def register_method(name: str):
    """Decorator to register a new FL method class."""

    def decorator(cls: Type[BaseMethod]) -> Type[BaseMethod]:
        cls.name = name
        METHOD_REGISTRY[name] = cls
        return cls

    return decorator


register_algorithm = register_method


def create_method(config: ExperimentConfig) -> BaseMethod:
    """Instantiate a registered FL method from `config.method`."""
    method_name = config.method
    if method_name not in METHOD_REGISTRY:
        supported = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(
            f"Unknown method '{method_name}'. Supported: {supported}"
        )
    return METHOD_REGISTRY[method_name](config)


create_algorithm = create_method

__all__ = [
    "ALGORITHM_REGISTRY",
    "BaseAlgorithm",
    "BaseMethod",
    "ClientResult",
    "FedAdamAlgorithm",
    "FedAdamMethod",
    "FedAvgAlgorithm",
    "FedAvgMAlgorithm",
    "FedAvgMMethod",
    "FedAvgMethod",
    "FedGPSAlgorithm",
    "FedGPSMethod",
    "FedLESAMAlgorithm",
    "FedLESAMMethod",
    "FedNovaAlgorithm",
    "FedNovaMethod",
    "FedProxAlgorithm",
    "FedProxMethod",
    "FedSpeedAlgorithm",
    "FedSpeedMethod",
    "FedWMSAMAlgorithm",
    "FedWMSAMMethod",
    "HeteroFLAlgorithm",
    "HeteroFLMethod",
    "METHOD_REGISTRY",
    "ScaffoldAlgorithm",
    "ScaffoldMethod",
    "StateDict",
    "create_algorithm",
    "create_method",
    "fedavg_aggregate",
    "register_algorithm",
    "register_method",
    "tensor_dict_norm",
    "zero_control",
]
