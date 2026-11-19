"""FedGPS (Yang et al., NeurIPS 2025) method package."""

from fl_bench.methods.fedgps.losses import (
    ProxyAlignLoss,
    SupConLoss,
    TripletLoss,
    local_proto_agg,
    proto_aggregation,
)
from fl_bench.methods.fedgps.method import FedGPSAlgorithm, FedGPSMethod
from fl_bench.methods.fedgps.optimizer import (
    FedGPSOptimizer,
    compute_fedgps_perturbation,
)
from fl_bench.methods.fedgps.surrogate import (
    FedGPSFeatureCatcher,
    build_fedgps_model,
    generate_surrogate_batch,
)

__all__ = [
    "FedGPSAlgorithm",
    "FedGPSFeatureCatcher",
    "FedGPSMethod",
    "FedGPSOptimizer",
    "ProxyAlignLoss",
    "SupConLoss",
    "TripletLoss",
    "build_fedgps_model",
    "compute_fedgps_perturbation",
    "generate_surrogate_batch",
    "local_proto_agg",
    "proto_aggregation",
]

