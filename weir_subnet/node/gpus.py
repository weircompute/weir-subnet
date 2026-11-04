"""What the node reports about its hardware, and how it solves challenges."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

from .. import challenge


@dataclass
class Gpu:
    index: int
    name: str
    uuid: str
    vram_mib: int


def detect() -> list[Gpu]:
    """GPUs as the NVIDIA driver reports them. Empty when there is no driver."""
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        index, name, uuid, mem = [part.strip() for part in line.split(",")]
        gpus.append(Gpu(int(index), name.removeprefix("NVIDIA "), uuid, int(float(mem))))
    return gpus


def simulated(spec: str, salt: str) -> list[Gpu]:
    """Development GPUs, e.g. "RTX 4090:24564:2". Solved on the CPU, so they
    only pass a validator that runs with development timing."""
    name, vram, count = spec.rsplit(":", 2)
    return [
        Gpu(i, name, "GPU-sim-" + hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:24], int(vram))
        for i in range(int(count))
    ]


def host_facts() -> dict:
    mem_mib = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_mib = int(line.split()[1]) // 1024
    except OSError:
        pass
    disk = shutil.disk_usage("/")
    return {"cpu_cores": os.cpu_count() or 0, "mem_mib": mem_mib, "disk_free_gib": disk.free // 2**30, "disk_total_gib": disk.total // 2**30}


def solve(spec: challenge.Spec, gpu: Gpu, simulated_gpu: bool) -> np.ndarray:
    if simulated_gpu:
        return challenge.solve_numpy(spec)
    return challenge.solve_torch(spec, f"cuda:{gpu.index}")


def as_json(gpus: list[Gpu]) -> list[dict]:
    return [asdict(g) for g in gpus]
