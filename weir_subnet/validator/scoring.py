"""Turning what the validator observed into weights.

The miner emission is split into three pools every time weights are set.

  rented  20%   shared by settled rental GPU-hours, each weighted by the
                GPU model's weight. Rent itself is paid separately (95% of
                what the tenant paid, in stablecoin), so this is a small extra.
  idle    <=80% paid per idle GPU that passes every gate below, at
                model weight x (80% / target_capacity) per GPU. Above a
                model's cap of idle GPUs, that model's pay is shared
                (cap / idle GPUs). If the total would exceed 80% it is scaled
                down to 80%.
  reserve rest  to the reserve hotkey (the owner UID when none is set): the
                network fund. It shrinks as providers fill the idle pool.

Idle gates, per machine:
  - the miner has activation credit (and, per GPU, that GPU's activation fee is paid);
  - the GPU passed a challenge within VERIFIED_TTL, is not rented, and has
    passed without a gap for the probation period;
  - the machine answered this round, availability >= 0.5, miner not struck;
  - its price per GPU-hour is at or below the model's published cap;
  - total disk >= 1.5 x total VRAM;
  - it runs tenants under Sysbox.
Disk and runtime are reported by the node; a node that lies there fails its
tenants, which shows up as refunds and penalties.

A wrong challenge answer zeroes the miner in every pool for 24 hours and
restarts its probation. A miner whose activation fee is unpaid earns nothing.

Rentals of a GPU make the rented pool, not the idle pool, so renting your own
GPU to yourself costs the platform fee and earns only a share of 20%.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

AVAIL_EMA = 0.1
VERIFIED_TTL = 6 * 3600  # a GPU pass counts for this long
PROBATION_SECONDS = 14 * 24 * 3600  # passes without a gap before idle pay starts
STRIKE_SECONDS = 24 * 3600  # zero score after a wrong answer


@dataclass
class GpuRecord:
    uuid: str
    model: str | None
    weight: float
    verified_at: float = 0.0
    last_result: str = ""  # human-readable detail
    node: str = "main"
    code: str = "PENDING"  # see codes.py
    index: int = 0
    verified_since: float = 0.0  # start of the current run of passes without a gap
    fee_covered: bool = True  # this GPU's activation fee is paid (set every round when a fee applies)


@dataclass
class NodeRecord:
    """One machine of a miner."""

    name: str
    url: str
    availability: float = 0.0
    probes: int = 0
    price: float | None = None
    code: str = "PENDING"
    detail: str = ""
    disk_total_gib: float = 0.0
    container_runtime: str = ""

    def record_probe(self, ok: bool) -> None:
        self.probes += 1
        value = 1.0 if ok else 0.0
        self.availability = value if self.probes == 1 else (1 - AVAIL_EMA) * self.availability + AVAIL_EMA * value


@dataclass
class MinerRecord:
    hotkey: str
    uid: int
    gpus: dict[str, GpuRecord] = field(default_factory=dict)  # by uuid
    availability: float = 0.0
    probes: int = 0
    price: float | None = None
    struck_until: float = 0.0
    note: str = ""
    last_challenge: float = 0.0
    payout_address: str = ""
    nodes: dict[str, NodeRecord] = field(default_factory=dict)  # by name
    fee_paid: bool = True  # has activation credit; set every round when a activation fee applies

    def refresh_totals(self) -> None:
        """Miner-level availability and price from its nodes."""
        if self.nodes:
            self.availability = sum(n.availability for n in self.nodes.values()) / len(self.nodes)
            prices = [n.price for n in self.nodes.values() if n.price]
            self.price = sum(prices) / len(prices) if prices else None

    def capacity(self, now: float) -> float:
        if now < self.struck_until:
            return 0.0
        return sum(g.weight for g in self.gpus.values() if g.model and now - g.verified_at < VERIFIED_TTL)

    def record_probe(self, ok: bool) -> None:
        self.probes += 1
        value = 1.0 if ok else 0.0
        self.availability = value if self.probes == 1 else (1 - AVAIL_EMA) * self.availability + AVAIL_EMA * value


@dataclass
class RentalStats:
    gpu_hours: float = 0.0
    delivery: float | None = None  # 0..1


@dataclass(frozen=True)
class Policy:
    rented_share: float = 0.20
    idle_ceiling: float = 0.80
    idle_cap_gpus: int = 64  # per model; above it, idle pay for that model is shared
    min_disk_ratio: float = 1.5  # total disk / total VRAM
    require_sysbox: bool = True
    sysbox_runtime: str = "sysbox-runc"
    probation_seconds: float = PROBATION_SECONDS


def _node_model(record: MinerRecord, node: str) -> str | None:
    counts: dict[str, int] = {}
    for g in record.gpus.values():
        if g.node == node and g.model:
            counts[g.model] = counts.get(g.model, 0) + 1
    return max(counts, key=counts.get) if counts else None


def price_gate(node: NodeRecord, record: MinerRecord, caps: dict[str, float]) -> str:
    """OK when the machine's price is within its model's cap."""
    from . import codes

    key = _node_model(record, node.name)
    if key is None:
        return codes.PENDING
    if key not in caps:
        return codes.NO_PRICE_CAP
    if not node.price or node.price <= 0 or node.price > caps[key] + 1e-9:
        return codes.PRICE_OVER_CAP
    return codes.OK


def in_probation(g: GpuRecord, now: float, policy: Policy) -> bool:
    return not g.verified_since or now - g.verified_since < policy.probation_seconds


def idle_gate(node: NodeRecord, record: MinerRecord, policy: Policy, caps: dict[str, float], now: float | None = None) -> str:
    """OK when this machine's idle GPUs earn idle pay, else the reason code."""
    from . import codes

    now = time.time() if now is None else now
    if not record.fee_paid:
        return codes.NOT_ACTIVATED
    if now < record.struck_until:
        return codes.STRUCK
    if node.code != codes.OK or node.availability < 0.5:
        return node.code if node.code != codes.OK else codes.UNREACHABLE
    price = price_gate(node, record, caps)
    if price != codes.OK:
        return price
    vram_gib = sum(_vram_gib(g) for g in record.gpus.values() if g.node == node.name and g.model)
    if policy.min_disk_ratio and node.disk_total_gib < policy.min_disk_ratio * vram_gib:
        return codes.DISK_TOO_SMALL
    if policy.require_sysbox and node.container_runtime != policy.sysbox_runtime:
        return codes.NO_SYSBOX
    return codes.OK


def _vram_gib(g: GpuRecord) -> float:
    from .models import MODELS

    model = next((m for m in MODELS if m.key == g.model), None)
    return model.vram_mib[0] / 1024 if model else 0.0


def compute_weights(
    miners: list[MinerRecord],
    rentals: dict[str, RentalStats],
    reserve_uid: int,
    target_capacity: float,
    now: float | None = None,
    policy: Policy = Policy(),
    caps: dict[str, float] | None = None,
) -> tuple[dict[int, float], dict[str, dict]]:
    """Returns ({uid: weight}, per-hotkey breakdown for the public summary)."""
    from . import codes
    from .models import MODELS

    now = time.time() if now is None else now
    caps = caps or {}
    active = [m for m in miners if now >= m.struck_until and m.fee_paid]

    # Rented pool.
    def model_weight(m: MinerRecord) -> float:
        weights = [g.weight for g in m.gpus.values() if g.model]
        return sum(weights) / len(weights) if weights else 1.0

    units = {}
    for m in active:
        stats = rentals.get(m.hotkey)
        if stats and stats.gpu_hours > 0:
            units[m.hotkey] = stats.gpu_hours * model_weight(m) * (stats.delivery if stats.delivery is not None else 1.0)
    total_units = sum(units.values())
    rented = {h: policy.rented_share * u / total_units for h, u in units.items()} if total_units > 0 else {}

    # Idle pool: GPUs that pass the gates, counted per model for the caps.
    idle_gpus: dict[str, list[tuple[str, float]]] = {}  # hotkey -> [(model, weight)]
    for m in active:
        for node in m.nodes.values():
            if idle_gate(node, m, policy, caps, now) != codes.OK:
                continue
            for g in m.gpus.values():
                if (g.node == node.name and g.model and g.fee_covered and g.code != codes.RENTED and now - g.verified_at < VERIFIED_TTL
                        and not in_probation(g, now, policy)):
                    idle_gpus.setdefault(m.hotkey, []).append((g.model, g.weight))
    per_model: dict[str, int] = {}
    for gpus in idle_gpus.values():
        for key, _ in gpus:
            per_model[key] = per_model.get(key, 0) + 1
    unit = policy.idle_ceiling / target_capacity if target_capacity > 0 else 0.0
    idle = {}
    for hotkey, gpus in idle_gpus.items():
        idle[hotkey] = sum(weight * unit * min(1.0, policy.idle_cap_gpus / per_model[key]) for key, weight in gpus)
    total_idle = sum(idle.values())
    if total_idle > policy.idle_ceiling:
        idle = {h: v * policy.idle_ceiling / total_idle for h, v in idle.items()}

    weights: dict[int, float] = {}
    detail = {}
    for m in miners:
        w = rented.get(m.hotkey, 0.0) + idle.get(m.hotkey, 0.0)
        if w > 0:
            weights[m.uid] = weights.get(m.uid, 0.0) + w
        detail[m.hotkey] = {"uid": m.uid, "rented": round(rented.get(m.hotkey, 0.0), 6), "idle": round(idle.get(m.hotkey, 0.0), 6),
                            "idle_gpus": len(idle_gpus.get(m.hotkey, [])), "weight": round(w, 6),
                            "nodes": {n.name: idle_gate(n, m, policy, caps, now) for n in m.nodes.values()},
                            "struck": now < m.struck_until, "activated": m.fee_paid, "note": m.note}
    rest = max(0.0, 1.0 - sum(weights.values()))
    if rest > 0:
        weights[reserve_uid] = weights.get(reserve_uid, 0.0) + rest
    return weights, detail
