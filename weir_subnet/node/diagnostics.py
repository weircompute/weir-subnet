"""Model and performance checks for the miner self-test."""

from __future__ import annotations

from dataclasses import dataclass

_RATE_PER_GBS = 5.1e9 / 1555  # elements/s per GB/s of memory bandwidth (A800 40GB)


@dataclass(frozen=True)
class Model:
    key: str
    names: tuple[str, ...]  # substrings of the driver name
    vram_mib: tuple[int, int]  # accepted range
    min_tflops: float  # slowest acceptable fp16 multiply rate
    mem_gbs: float  # memory bandwidth, GB/s (the lower variant where SXM and PCIe share a name)
    calibrated: bool = False

    @property
    def elem_rate(self) -> float:
        return self.mem_gbs * _RATE_PER_GBS


MODELS: tuple[Model, ...] = (
    Model("h100-80g", ("H100",), (79_000, 82_000), 250.0, 2000),
    Model("a100-80g", ("A100",), (79_000, 82_000), 200.0, 1935),
    Model("a100-40g", ("A100",), (39_000, 41_000), 200.0, 1555),
    Model("a800-80g", ("A800",), (79_000, 82_000), 200.0, 1935),
    Model("a800-40g", ("A800",), (39_000, 41_000), 200.0, 1555, calibrated=True),
    Model("rtx5090-32g", ("RTX 5090",), (31_000, 33_000), 130.0, 1792),
    Model("rtx4090-48g", ("RTX 4090",), (48_000, 50_000), 100.0, 1008),
    Model("rtx4090-24g", ("RTX 4090",), (23_500, 24_600), 100.0, 1008),
)


TIMING_SLACK = 1.3
UNCALIBRATED_SLACK = 1.5  # extra factor for models not yet measured
TIMING_OVERHEAD_S = 3.0  # CUDA start-up, copying results, network


def expected_seconds(flops: float, elements: float, model: Model) -> float:
    return elements / model.elem_rate + flops / (model.min_tflops * 1e12)


def time_limit(flops: float, elements: float, model: Model, slack: float = TIMING_SLACK, overhead_s: float = TIMING_OVERHEAD_S) -> float:
    """Self-test time limit for this model."""
    factor = slack * (1.0 if model.calibrated else UNCALIBRATED_SLACK)
    return factor * expected_seconds(flops, elements, model) + overhead_s


def classify(name: str, vram_mib: int) -> Model | None:
    for model in MODELS:
        if any(part in name for part in model.names) and model.vram_mib[0] <= vram_mib <= model.vram_mib[1]:
            return model
    return None
