"""APT: deterministic local-first cognitive systems research substrate."""

from .memory import ContentAddressedMemory
from .metrics import CognitiveMetrics, MetricReport
from .precision import DeterministicPrecisionKernel, NumericalContract
from .simulator import Hypothesis, SimulatorBank
from .system import CognitiveOrganism
from .timeline import TimelineDAG

__all__ = [
    "CognitiveMetrics",
    "CognitiveOrganism",
    "ContentAddressedMemory",
    "DeterministicPrecisionKernel",
    "Hypothesis",
    "MetricReport",
    "NumericalContract",
    "SimulatorBank",
    "TimelineDAG",
]

__version__ = "1.0.0"

