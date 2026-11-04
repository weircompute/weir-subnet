"""GPU proof-of-work challenge shared by the node (miner) and the validator.

The validator sends a random seed. The node builds `count` pairs of n x n
matrices from the seed, multiplies each pair on the GPU, and returns a
commitment (sha256 over one fingerprint per output row). The validator then
picks random rows; the node reveals those rows and the full fingerprint list.
The validator recomputes the rows on its CPU and checks them against the
commitment.

Why this works:
- The commitment arrives before the validator says which rows it will check,
  so a node that skipped part of the work is caught with high probability.
- The fingerprint is non-linear in the row values, so it cannot be derived
  from cheaper products such as A @ (B @ w).
- Matrix entries are in {-1, 0, 1}, so every product row sum stays far below
  2048 and fp16 matmul is exact. GPU and CPU results match bit for bit.
- The validator times the commitment. Only a card of the claimed class
  finishes inside the time limit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# 64-bit mixing constants (splitmix64), written as signed int64 so numpy and
# torch both wrap the same way.
_M1 = 0x9E3779B97F4A7C15 - (1 << 64)
_M2 = 0xBF58476D1CE4E5B9 - (1 << 64)
_M3 = 0x94D049BB133111EB - (1 << 64)
_SALT_A = 0
_SALT_B = 1

PROTOCOL = "weir-mm/1"


@dataclass(frozen=True)
class Spec:
    """One challenge for one GPU."""

    challenge_id: str
    seed: int  # 0 <= seed < 2**62
    n: int  # matrix side
    count: int  # number of matrix pairs

    def to_json(self) -> dict:
        return {"protocol": PROTOCOL, "challenge_id": self.challenge_id, "seed": self.seed, "n": self.n, "count": self.count}

    @staticmethod
    def from_json(data: dict) -> "Spec":
        if data.get("protocol") != PROTOCOL:
            raise ValueError("unknown challenge protocol")
        spec = Spec(str(data["challenge_id"]), int(data["seed"]), int(data["n"]), int(data["count"]))
        if not (0 <= spec.seed < 2**62 and 64 <= spec.n <= 32768 and spec.n % 64 == 0 and 1 <= spec.count <= 4096):
            raise ValueError("challenge out of range")
        return spec

    @property
    def flops(self) -> float:
        return 2.0 * self.count * self.n**3

    @property
    def elements(self) -> float:
        """Elements generated or fingerprinted: A, B and C for every pair."""
        return 3.0 * self.count * self.n**2


# ---------------------------------------------------------------------------
# Element generation. The same integer arithmetic runs in numpy (validator,
# CPU node) and torch (GPU node). Both use int64 with wrap-around and an
# arithmetic right shift.


def _mix_np(x: np.ndarray) -> np.ndarray:
    x = x * np.int64(_M1)
    x = x ^ (x >> np.int64(31))
    x = x * np.int64(_M2)
    x = x ^ (x >> np.int64(29))
    return x


def _ternary_np(h: np.ndarray) -> np.ndarray:
    # magnitude bit and sign bit -> {-1, 0, 1}, P(0) = 1/2
    mag = (h >> np.int64(7)) & np.int64(1)
    sign = ((h >> np.int64(13)) & np.int64(1)) * np.int64(2) - np.int64(1)
    return mag * sign


def _base(index: int, salt: int) -> int:
    # Each matrix gets its own 2**31 slot (n*n <= 2**30), so no two matrices
    # share elements. The seed is folded in with XOR before mixing.
    return (index * 2 + salt) << 31


def rows_np(spec: Spec, index: int, salt: int, row_start: int, row_stop: int) -> np.ndarray:
    """Rows [row_start, row_stop) of matrix A (salt 0) or B (salt 1), as int64."""
    n = spec.n
    i = np.arange(row_start, row_stop, dtype=np.int64)[:, None]
    j = np.arange(n, dtype=np.int64)[None, :]
    idx = (i * np.int64(n) + j + np.int64(_base(index, salt))) ^ np.int64(spec.seed)
    with np.errstate(over="ignore"):
        return _ternary_np(_mix_np(idx))


def fingerprint_np(rows: np.ndarray, row_ids: np.ndarray) -> np.ndarray:
    """One int64 fingerprint per row of the product C."""
    n = rows.shape[1]
    j = np.arange(n, dtype=np.int64)[None, :]
    with np.errstate(over="ignore"):
        x = rows.astype(np.int64) * np.int64(1 << 20) + j * np.int64(2654435761) + row_ids.astype(np.int64)[:, None] * np.int64(_M3)
        return _mix_np(x).sum(axis=1, dtype=np.int64)


def commitment(spec: Spec, fingerprints: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{PROTOCOL}|{spec.challenge_id}|{spec.seed}|{spec.n}|{spec.count}|".encode())
    h.update(np.ascontiguousarray(fingerprints, dtype="<i8").tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Solvers (node side).


def solve_numpy(spec: Spec) -> np.ndarray:
    """Reference solver. Slow; for tests and CPU-only development nodes."""
    out = np.empty((spec.count, spec.n), dtype=np.int64)
    rows = np.arange(spec.n, dtype=np.int64)
    for b in range(spec.count):
        a = rows_np(spec, b, _SALT_A, 0, spec.n)
        bm = rows_np(spec, b, _SALT_B, 0, spec.n)
        c = a.astype(np.float32) @ bm.astype(np.float32)  # exact: |c| < 2**24
        out[b] = fingerprint_np(c.astype(np.int64), rows)
    return out


def solve_torch(spec: Spec, device: str = "cuda:0") -> np.ndarray:
    """GPU solver. All A and B matrices are built in GPU memory first, then
    multiplied in fp16. Returns the fingerprints as a numpy array."""
    import torch

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dev = torch.device(device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    n = spec.n

    def mix(x):
        x = x * _M1
        x = x ^ (x >> 31)
        x = x * _M2
        x = x ^ (x >> 29)
        return x

    # One block for all A and B, allocated before any scratch, and scratch
    # of a fixed size: varying sizes fragmented memory and ran out at n=6144.
    a_all = torch.empty((spec.count, n, n), dtype=dtype, device=dev)
    b_all = torch.empty((spec.count, n, n), dtype=dtype, device=dev)

    def build(index: int, salt: int):
        out = (a_all if salt == _SALT_A else b_all)[index]
        step = max(1, (1 << 24) // n)  # bound int64 scratch to ~128 MB
        j = torch.arange(n, dtype=torch.int64, device=dev)[None, :]
        base = _base(index, salt)
        for r0 in range(0, n, step):
            i = torch.arange(r0, min(n, r0 + step), dtype=torch.int64, device=dev)[:, None]
            h = mix((i * n + j + base) ^ spec.seed)
            mag = (h >> 7) & 1
            sign = ((h >> 13) & 1) * 2 - 1
            out[r0 : r0 + step] = (mag * sign).to(dtype)
        return out

    with torch.no_grad():
        for b in range(spec.count):
            build(b, _SALT_A)
            build(b, _SALT_B)
        a, bm = a_all, b_all
        jj = torch.arange(n, dtype=torch.int64, device=dev)[None, :]
        rows = torch.arange(n, dtype=torch.int64, device=dev)[:, None]
        fps = torch.empty((spec.count, n), dtype=torch.int64, device=dev)
        for b in range(spec.count):
            c = torch.matmul(a[b], bm[b]).to(torch.int64)
            x = c * (1 << 20) + jj * 2654435761 + rows * _M3
            fps[b] = mix(x).sum(dim=1)
        return fps.cpu().numpy()


def product_rows(spec: Spec, index: int, row_ids: list[int], chunk: int = 512) -> np.ndarray:
    """Rows `row_ids` of C = A @ B for one matrix pair, on the CPU, building B
    a few hundred rows at a time so memory stays around 100 MB."""
    a = np.concatenate([rows_np(spec, index, _SALT_A, i, i + 1) for i in row_ids])
    out = np.zeros((len(row_ids), spec.n), dtype=np.int64)
    for k0 in range(0, spec.n, chunk):
        k1 = min(spec.n, k0 + chunk)
        out += a[:, k0:k1] @ rows_np(spec, index, _SALT_B, k0, k1)
    return out


def reveal_rows(spec: Spec, picks: list[tuple[int, int]]) -> list[list[int]]:
    """The product rows the validator asked for."""
    return [product_rows(spec, b, [i])[0].tolist() for b, i in picks]


# ---------------------------------------------------------------------------
# Verification (validator side).


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


def verify(spec: Spec, root: str, fingerprints: list[int], picks: list[tuple[int, int]], rows: list[list[int]]) -> Verdict:
    fps = np.asarray(fingerprints, dtype=np.int64)
    if fps.shape != (spec.count * spec.n,):
        return Verdict(False, "fingerprint list has the wrong length")
    fps = fps.reshape(spec.count, spec.n)
    if commitment(spec, fps) != root:
        return Verdict(False, "fingerprints do not match the commitment")
    if len(rows) != len(picks):
        return Verdict(False, "wrong number of revealed rows")
    wanted: dict[int, list[int]] = {}
    for b, i in picks:
        if not (0 <= b < spec.count and 0 <= i < spec.n):
            return Verdict(False, "pick out of range")
        wanted.setdefault(b, []).append(i)
    expected = {b: dict(zip(ids, product_rows(spec, b, ids))) for b, ids in wanted.items()}
    for (b, i), row in zip(picks, rows):
        want = expected[b][i]
        got = np.asarray(row, dtype=np.int64)
        if got.shape != want.shape or not np.array_equal(got, want):
            return Verdict(False, f"row {b}:{i} is wrong")
        if fingerprint_np(want[None, :], np.array([i]))[0] != fps[b, i]:
            return Verdict(False, f"row {b}:{i} does not match its fingerprint")
    return Verdict(True)


def pick_rows(spec: Spec, k: int, rng) -> list[tuple[int, int]]:
    """k rows to check, each from a different matrix pair where possible, so
    a node that skipped a share of the pairs is caught as often as possible:
    with k = 4 and a quarter of the pairs skipped, 1 - (3/4)^4 = 68% per
    challenge, and challenges repeat every hour."""
    pairs = rng.sample(range(spec.count), min(k, spec.count))
    while len(pairs) < k:
        pairs.append(rng.randrange(spec.count))
    return sorted({(b, rng.randrange(spec.n)) for b in pairs})


def size_for(vram_mib: int, n: int = 8192, fill: float = 0.7) -> int:
    """Matrix-pair count that fills `fill` of the card's memory (A and B in fp16)."""
    per_pair = 2 * 2 * n * n
    return max(1, int(vram_mib * 1024 * 1024 * fill // per_pair))
