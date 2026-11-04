"""GPU models the subnet accepts, and how long a challenge may take on each.

A challenge has two kinds of work: generating and fingerprinting matrix
elements (64-bit integer hashing, bound by memory bandwidth) and the fp16
matrix multiply (tensor cores). Measured on an A800 SXM4 40GB at n=8192,
112 pairs: generation 3.4 s, fingerprints 1.0 s, multiply 0.4 s (294 TFLOPS),
end to end 6.6-7.1 s including CUDA start-up and copying results back. So
the limit counts both, and generation dominates.

elem_rate is elements per second for generation + fingerprints (3 elements
per pair cell: A, B and C). Measured on the A800 40GB: 22.5e9 elements in
4.4 s, about 5.1e9/s at 1555 GB/s. Other models are scaled by memory
bandwidth and marked calibrated=False: their limits get extra slack until
someone runs `weir-node selftest --json` on that card.

weight is the model's relative value when sharing out emissions, roughly in
line with what each card rents for. fee_tao is the one-time activation fee for
one GPU of the model. Both are defaults: the platform file can set others
per model ("weights", "fees"), so they can change without a release.
"""

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
    weight: float
    fee_tao: float
    calibrated: bool = False

    @property
    def elem_rate(self) -> float:
        return self.mem_gbs * _RATE_PER_GBS


MODELS: tuple[Model, ...] = (
    Model("h100-80g", ("H100",), (79_000, 82_000), 250.0, 2000, 4.0, 10.0),
    Model("a100-80g", ("A100",), (79_000, 82_000), 200.0, 1935, 2.5, 7.8),
    Model("a100-40g", ("A100",), (39_000, 41_000), 200.0, 1555, 1.8, 6.1),
    Model("a800-80g", ("A800",), (79_000, 82_000), 200.0, 1935, 2.4, 7.6),
    Model("a800-40g", ("A800",), (39_000, 41_000), 200.0, 1555, 1.7, 5.9, calibrated=True),
    Model("rtx5090-32g", ("RTX 5090",), (31_000, 33_000), 130.0, 1792, 1.3, 4.7),
    Model("rtx4090-48g", ("RTX 4090",), (48_000, 50_000), 100.0, 1008, 1.2, 4.4),
    Model("rtx4090-24g", ("RTX 4090",), (23_500, 24_600), 100.0, 1008, 1.0, 3.7),
)


TIMING_SLACK = 1.3
UNCALIBRATED_SLACK = 1.5  # extra factor for models not yet measured
TIMING_OVERHEAD_S = 3.0  # CUDA start-up, copying results, network


def expected_seconds(flops: float, elements: float, model: Model) -> float:
    return elements / model.elem_rate + flops / (model.min_tflops * 1e12)


def time_limit(flops: float, elements: float, model: Model, slack: float = TIMING_SLACK, overhead_s: float = TIMING_OVERHEAD_S) -> float:
    """Seconds a challenge may take on `model`. The validator and the miner
    self-test use this same rule. A800 40GB, n=8192: limit 9.3 s, measured 6.6-7.1 s."""
    factor = slack * (1.0 if model.calibrated else UNCALIBRATED_SLACK)
    return factor * expected_seconds(flops, elements, model) + overhead_s


def classify(name: str, vram_mib: int) -> Model | None:
    for model in MODELS:
        if any(part in name for part in model.names) and model.vram_mib[0] <= vram_mib <= model.vram_mib[1]:
            return model
    return None
