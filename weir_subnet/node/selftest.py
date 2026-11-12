"""GPU self-test."""

from __future__ import annotations

import asyncio
import json
import random
import secrets
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field

from .. import challenge
from .diagnostics import classify, time_limit
from . import gpus as gpu_mod


@dataclass
class Check:
    name: str
    result: str  # PASS | WARN | FAIL
    detail: str = ""
    measured: dict = field(default_factory=dict)


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, str(error)


def _challenge_gpu(gpu: gpu_mod.Gpu, n: int, simulated: bool, timing: bool) -> Check:
    name = f"GPU {gpu.index} {gpu.name}"
    model = classify(gpu.name, gpu.vram_mib)
    if not model:
        return Check(name, "FAIL", f"{gpu.name} with {gpu.vram_mib} MiB is not an accepted model")
    spec = challenge.Spec(secrets.token_hex(8), secrets.randbits(62), n, challenge.size_for(gpu.vram_mib, n))
    if simulated:
        spec = challenge.Spec(spec.challenge_id, spec.seed, n, min(spec.count, 2))
    started = time.perf_counter()
    fps = gpu_mod.solve(spec, gpu, simulated)
    seconds = time.perf_counter() - started
    picks = challenge.pick_rows(spec, 4, random.SystemRandom())
    verdict = challenge.verify(spec, challenge.commitment(spec, fps), fps.ravel().tolist(), picks, challenge.reveal_rows(spec, picks))
    limit = time_limit(spec.flops, spec.elements, model)
    measured = {"model": model.key, "n": spec.n, "pairs": spec.count, "seconds": round(seconds, 3),
                "limit_seconds": round(limit, 3), "tflops": round(spec.flops / seconds / 1e12, 2)}
    if not verdict.ok:
        return Check(name, "FAIL", f"wrong result: {verdict.reason}", measured)
    if timing and seconds > limit:
        return Check(name, "FAIL", f"{seconds:.1f}s, limit {limit:.1f}s: check power limit, clocks and other load", measured)
    return Check(name, "PASS", f"{seconds:.1f}s of {limit:.1f}s allowed, {measured['tflops']} TFLOPS", measured)


async def run(gpu_list: list[gpu_mod.Gpu], n: int = 8192, simulated: bool = False, image: str = "",
              docker_runtime: str = "", payout_address: str = "") -> list[Check]:
    checks: list[Check] = []
    if not gpu_list:
        return [Check("GPUs", "FAIL", "no GPUs found (is the NVIDIA driver installed?)")]
    uuids = [g.uuid for g in gpu_list]
    checks.append(Check("GPUs", "PASS" if len(set(uuids)) == len(uuids) else "FAIL", f"{len(gpu_list)} found"))
    # All GPUs at once, as the validator does.
    results = await asyncio.gather(*(asyncio.to_thread(_challenge_gpu, g, n, simulated, not simulated) for g in gpu_list))
    checks.extend(results)

    if not shutil.which("docker"):
        checks.append(Check("Docker", "FAIL", "docker is not installed: tenants' machines run in containers"))
    else:
        code, out = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
        runtimes = json.loads(out) if code == 0 and out.startswith("{") else {}
        checks.append(Check("Docker", "PASS" if code == 0 else "FAIL", "running" if code == 0 else out[-200:]))
        if not simulated:
            checks.append(Check("NVIDIA container runtime", "PASS" if "nvidia" in runtimes else "FAIL",
                                "found" if "nvidia" in runtimes else "install nvidia-container-toolkit"))
        wanted = docker_runtime or "sysbox-runc"
        checks.append(Check("Sysbox", "PASS" if wanted in runtimes else "WARN",
                            "found: tenants can run Docker inside" if wanted in runtimes else "not installed: tenants cannot run Docker inside their machine"))
        if image and code == 0:
            cmd = ["docker", "run", "--rm"] + ([] if simulated else ["--gpus", "all"]) + ["--entrypoint", "sh", image, "-c",
                   "nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0"]
            if wanted in runtimes:
                cmd[3:3] = ["--runtime", wanted]
            rc, out = _run(cmd, timeout=300)
            seen = int(out.split()[-1]) if rc == 0 and out.split() and out.split()[-1].isdigit() else -1
            expected = 0 if simulated else len(gpu_list)
            ok = rc == 0 and seen == expected
            checks.append(Check("GPUs inside a container", "PASS" if ok else "FAIL",
                                f"{seen} of {expected} visible in {image}" if rc == 0 else out[-200:]))
    if payout_address:
        ok = payout_address.isalnum() and 20 <= len(payout_address) <= 100
        checks.append(Check("Payout address", "PASS" if ok else "FAIL", payout_address if ok else "does not look like a wallet address"))
    else:
        checks.append(Check("Payout address", "WARN", "not set: earnings are kept until you set --payout-address"))
    return checks


def report(checks: list[Check], as_json: bool) -> str:
    if as_json:
        return json.dumps([asdict(c) for c in checks], indent=1)
    width = max(len(c.name) for c in checks)
    lines = [f"{c.result:4}  {c.name:<{width}}  {c.detail}" for c in checks]
    failed = sum(c.result == "FAIL" for c in checks)
    lines.append("" if not failed else f"\n{failed} check(s) failed. The validator will not sell GPUs that fail.")
    return "\n".join(lines).rstrip()
