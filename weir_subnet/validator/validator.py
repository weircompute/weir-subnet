"""The Weir validator loop.

Each round:
1. read the metagraph;
2. probe every miner that publishes an endpoint (inventory, signed);
3. challenge GPUs whose verification is missing or getting old, all GPUs of
   a node at the same time so one real card cannot stand in for several;
4. read activation fees paid to the fee address from the chain;
5. every `weights_every` seconds, turn the records into weights.

State is a JSON file so a restart keeps verification history.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import os
import random
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
from bittensor_core import Keypair

from .. import challenge
from .chain import Chain, Neuron, Snapshot
from .client import NodeClient, NodeError
from . import codes
from .models import MODELS, TIMING_OVERHEAD_S, TIMING_SLACK, classify, time_limit
from .scoring import (STRIKE_SECONDS, VERIFIED_TTL, GpuRecord, MinerRecord, NodeRecord, Policy, RentalStats, compute_weights,
                      idle_gate, in_probation, price_gate)

MAX_NODES_PER_MINER = 64
MAX_SCAN_BLOCKS = 600  # blocks of fee history read per round while catching up
RAO_PER_TAO = 10**9

log = logging.getLogger("weir.validator")


@dataclass
class ValidatorConfig:
    state_file: Path
    challenge_n: int = 0  # fixed size (tests); 0 picks one of challenge_sizes per challenge
    challenge_sizes: tuple[int, ...] = (6144, 7168, 8192)
    fill: float = 0.7  # share of VRAM the challenge occupies
    max_pairs: int = 4096
    picks_per_gpu: int = 4
    recheck_after: float = 3600.0  # re-challenge a GPU after this long (with jitter)
    target_capacity: float = 1134.0  # idle weighted GPUs that fill the idle pool
    policy: Policy = dataclasses.field(default_factory=Policy)
    weights_every: float = 1200.0
    timing: bool = True  # False only for development nodes with simulated GPUs
    timing_slack: float = TIMING_SLACK
    timing_overhead_s: float = TIMING_OVERHEAD_S
    rentals_file: Path | None = None
    summary_file: Path | None = None
    sellable_file: Path | None = None  # nodes and verified GPUs the platform may sell
    status_file: Path | None = None  # per node and GPU reason codes, for miners
    platform_file: Path | None = None  # price caps, weight and fee overrides, activation credit paid through the platform
    reserve_hotkey: str = ""  # receives what the pools leave; the owner UID when unset
    fee_address: str = ""  # coldkey that receives activation fees; empty: no fee
    fee_start_block: int = 0  # first block to read fees from; 0: from the first round
    scheme: str = "http"


class Validator:
    def __init__(self, keypair: Keypair, chain: Chain, config: ValidatorConfig):
        self.keypair = keypair
        self.chain = chain
        self.config = config
        self.miners: dict[str, MinerRecord] = {}
        self.last_weights = 0.0
        self.fee_scanned_block = 0
        self.fees_paid: dict[str, int] = {}  # coldkey -> rao sent to the fee address
        self.covered: dict[str, dict] = {}  # GPU uuid -> {hotkey, coldkey, rao}: activation fees spent, kept for good
        self.credit_left: dict[str, int] = {}  # coldkey -> unspent activation credit in rao, after the last round
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.config.state_file.exists():
            return
        raw = json.loads(self.config.state_file.read_text())
        for m in raw.get("miners", []):
            gpus = {u: GpuRecord(**g) for u, g in m.pop("gpus").items()}
            nodes = {n: NodeRecord(**r) for n, r in m.pop("nodes", {}).items()}
            self.miners[m["hotkey"]] = MinerRecord(gpus=gpus, nodes=nodes, **m)
        self.last_weights = raw.get("last_weights", 0.0)
        self.fee_scanned_block = int(raw.get("fee_scanned_block", 0))
        self.fees_paid = {k: int(v) for k, v in raw.get("fees_paid", {}).items()}
        self.covered = dict(raw.get("covered", {}))

    def _save(self) -> None:
        self._write(self.config.state_file, {"miners": [dataclasses.asdict(m) for m in self.miners.values()], "last_weights": self.last_weights,
                                             "fee_scanned_block": self.fee_scanned_block, "fees_paid": self.fees_paid,
                                             "covered": self.covered})

    @staticmethod
    def _write(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, path)

    def _rentals(self) -> dict[str, RentalStats]:
        path = self.config.rentals_file
        if not path or not path.exists():
            return {}
        # Written by weir-api's `weir-payouts rentals`: billed GPU-hours and
        # the share of orders delivered without a penalty.
        return {hotkey: RentalStats(float(v.get("gpu_hours", 0)), v.get("delivery"))
                for hotkey, v in json.loads(path.read_text()).items()}

    def _platform(self) -> dict:
        path = self.config.platform_file
        if not path or not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except ValueError:
            log.warning("platform file is not JSON")
            return {}

    def _caps(self) -> dict[str, float]:
        """Highest price per GPU-hour per model, from the platform file."""
        caps = {}
        for key, value in self._platform().get("caps", {}).items():
            try:
                if float(value) > 0:
                    caps[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return caps

    def _per_model(self, name: str) -> dict[str, float]:
        """Per-model numbers from the platform file ("weights" or "fees"), over the defaults."""
        out = {m.key: (m.weight if name == "weights" else m.fee_tao) for m in MODELS}
        given = self._platform().get(name, {})
        for key, value in (given.items() if isinstance(given, dict) else ()):
            try:
                if key in out and float(value) >= 0:
                    out[key] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def _credit_elsewhere(self) -> dict[str, float]:
        """Activation fees paid through the platform (stablecoins), in TAO per hotkey."""
        given = self._platform().get("activation_credit", {})
        out = {}
        for hotkey, value in (given.items() if isinstance(given, dict) else ()):
            try:
                if float(value) > 0:
                    out[str(hotkey)] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def _special(self, snapshot: Snapshot) -> set[str]:
        """Hotkeys that are not scored as ordinary miners."""
        return {h for h in (snapshot.owner_hotkey, self.keypair.ss58_address, self.config.reserve_hotkey) if h}

    def _reserve_uid(self, snapshot: Snapshot) -> int:
        if self.config.reserve_hotkey:
            for n in snapshot.neurons:
                if n.hotkey == self.config.reserve_hotkey:
                    return n.uid
            log.warning("reserve hotkey is not registered: the rest goes to the owner UID")
        return snapshot.owner_uid

    # -- activation fees --------------------------------------------------------

    async def _scan_fees(self, snapshot: Snapshot) -> None:
        """Adds transfers to the fee address since the last scan."""
        if not self.config.fee_address:
            return
        if not self.fee_scanned_block:
            self.fee_scanned_block = (self.config.fee_start_block or snapshot.block) - 1
        start, end = self.fee_scanned_block + 1, min(snapshot.block, self.fee_scanned_block + MAX_SCAN_BLOCKS)
        if start > end:
            return
        try:
            transfers = await asyncio.to_thread(self.chain.transfers_to, self.config.fee_address, start, end)
        except Exception as error:
            log.warning("fee scan %d-%d failed: %s", start, end, error)
            return
        for sender, rao, _block in transfers:
            self.fees_paid[sender] = self.fees_paid.get(sender, 0) + int(rao)
        self.fee_scanned_block = end

    def _apply_fees(self, snapshot: Snapshot) -> None:
        """Every listed GPU needs its own activation fee, set per model. Credit is
        TAO sent from a coldkey to the fee address plus what its hotkeys paid
        through the platform. Credit goes to GPUs that passed a challenge,
        most valuable model first; a GPU once covered stays covered. A miner
        is in good standing while it has a covered GPU or credit for one."""
        if not self.config.fee_address:
            for record in self.miners.values():
                record.fee_paid = True
                for g in record.gpus.values():
                    g.fee_covered = True
            return
        fees = {k: round(v * RAO_PER_TAO) for k, v in self._per_model("fees").items()}
        cheapest = min(fees.values(), default=0)
        special = self._special(snapshot)
        coldkey_of = {n.hotkey: n.coldkey for n in snapshot.neurons if n.hotkey not in special}
        credit: dict[str, int] = dict(self.fees_paid)
        for hotkey, tao in self._credit_elsewhere().items():
            if hotkey in coldkey_of:
                credit[coldkey_of[hotkey]] = credit.get(coldkey_of[hotkey], 0) + round(tao * RAO_PER_TAO)
        for entry in self.covered.values():
            credit[entry["coldkey"]] = credit.get(entry["coldkey"], 0) - int(entry["rao"])
        candidates = []
        for record in self.miners.values():
            coldkey = coldkey_of.get(record.hotkey)
            for g in record.gpus.values():
                entry = self.covered.get(g.uuid)
                g.fee_covered = bool(coldkey and entry and entry["coldkey"] == coldkey)
                if coldkey and not entry and g.model and g.verified_since:
                    candidates.append((record, coldkey, g))
        for record, coldkey, g in sorted(candidates, key=lambda c: (-c[2].weight, c[0].uid, c[2].index)):
            fee = fees.get(g.model, 0)
            if credit.get(coldkey, 0) >= fee:
                credit[coldkey] = credit.get(coldkey, 0) - fee
                self.covered[g.uuid] = {"hotkey": record.hotkey, "coldkey": coldkey, "rao": fee}
                g.fee_covered = True
        self.credit_left = credit
        for record in self.miners.values():
            coldkey = coldkey_of.get(record.hotkey)
            record.fee_paid = bool(coldkey) and (any(g.fee_covered for g in record.gpus.values())
                                                 or credit.get(coldkey, 0) >= cheapest)

    # -- one round ---------------------------------------------------------

    async def run_round(self, now: float | None = None) -> dict[int, float] | None:
        snapshot = await asyncio.to_thread(self.chain.snapshot)
        special = self._special(snapshot)
        miners = [n for n in snapshot.neurons if n.axon and n.hotkey not in special]
        self._forget_departed(snapshot)
        for hotkey in special:
            self.miners.pop(hotkey, None)
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(self._visit_miner(session, n) for n in miners))
        self._check_duplicates()
        await self._scan_fees(snapshot)
        self._apply_fees(snapshot)
        self._save()
        self._write_sellable(snapshot)
        self._write_status(snapshot)
        now = time.time() if now is None else now
        if now - self.last_weights >= self.config.weights_every:
            return await self._set_weights(snapshot, now)
        return None

    def _forget_departed(self, snapshot: Snapshot) -> None:
        current = {n.hotkey: n.uid for n in snapshot.neurons}
        for hotkey in list(self.miners):
            if hotkey not in current:
                del self.miners[hotkey]  # deregistered: history goes with the UID
            else:
                self.miners[hotkey].uid = current[hotkey]

    async def _fleet(self, session: aiohttp.ClientSession, neuron: Neuron) -> list[tuple[str, str]] | None:
        """The miner's machines: (name, url). One machine unless the node at the
        chain address lists more. None when that node cannot be reached."""
        axon_url = f"{self.config.scheme}://{neuron.axon}"
        client = NodeClient(session, self.keypair, axon_url, neuron.hotkey)
        reply = await client.call("GET", "/v1/fleet", timeout=15)
        if reply.status == 404:
            return [("main", axon_url)]
        if reply.status != 200:
            raise NodeError(f"fleet HTTP {reply.status}")
        fleet = []
        for entry in reply.data.get("nodes", [])[:MAX_NODES_PER_MINER]:
            name, url = str(entry.get("name", "")), str(entry.get("url", ""))
            if name.isidentifier() and url.startswith(("http://", "https://")) and name not in {n for n, _ in fleet}:
                fleet.append((name, url.rstrip("/")))
        return fleet or [("main", axon_url)]

    async def _visit_miner(self, session: aiohttp.ClientSession, neuron: Neuron) -> None:
        record = self.miners.setdefault(neuron.hotkey, MinerRecord(neuron.hotkey, neuron.uid))
        try:
            fleet = await self._fleet(session, neuron)
        except NodeError as error:
            # The node at the chain address is down: every machine counts as missed.
            if not record.nodes:
                record.nodes["main"] = NodeRecord("main", f"{self.config.scheme}://{neuron.axon}")
            for node in record.nodes.values():
                node.record_probe(False)
                node.code, node.detail = self._error_code(error), str(error)
            record.refresh_totals()
            record.note = str(error)
            return
        names = {name for name, _ in fleet}
        for gone in [n for n in record.nodes if n not in names]:
            del record.nodes[gone]
        for uuid in [u for u, g in record.gpus.items() if g.node not in names]:
            del record.gpus[uuid]
        for name, url in fleet:
            node = record.nodes.get(name)
            if node is None or node.url != url:
                record.nodes[name] = NodeRecord(name, url)
        await asyncio.gather(*(self._visit_node(session, record, record.nodes[name]) for name, _ in fleet))
        record.refresh_totals()
        record.note = "; ".join(f"{n.name}: {n.code}" for n in record.nodes.values() if n.code not in (codes.OK, codes.PENDING))

    @staticmethod
    def _error_code(error: NodeError) -> str:
        return codes.BAD_SIGNATURE if "signed" in str(error) else codes.UNREACHABLE

    async def _visit_node(self, session: aiohttp.ClientSession, record: MinerRecord, node: NodeRecord) -> None:
        client = NodeClient(session, self.keypair, node.url, record.hotkey)
        try:
            reply = await client.call("GET", "/v1/inventory", timeout=15)
            if reply.status != 200:
                node.record_probe(False)
                node.code, node.detail = codes.HTTP_ERROR, f"inventory HTTP {reply.status}"
                return
        except NodeError as error:
            node.record_probe(False)
            node.code, node.detail = self._error_code(error), str(error)
            return
        if str(reply.data.get("name", "main")) != node.name:
            node.record_probe(False)
            node.code, node.detail = codes.FLEET_INVALID, f"node calls itself {reply.data.get('name')!r}"
            return
        node.record_probe(True)
        node.code, node.detail = codes.OK, ""
        self._update_inventory(record, node, reply.data)
        due = [g for g in reply.data.get("gpus", []) if self._due(record, g)]
        if time.time() < record.struck_until:
            node.code = codes.STRUCK
        elif due:
            record.last_challenge = time.time()
            await asyncio.gather(*(self._challenge(client, record, g) for g in due))

    def _update_inventory(self, record: MinerRecord, node: NodeRecord, inventory: dict) -> None:
        mine = {u: g for u, g in record.gpus.items() if g.node == node.name}
        weights = self._per_model("weights")
        reported = {}
        for g in inventory.get("gpus", []):
            model = classify(str(g.get("name", "")), int(g.get("vram_mib", 0)))
            uuid = str(g.get("uuid", ""))
            key = model.key if model else None
            old = mine.get(uuid)
            keep = old is not None and old.model == key
            gpu = GpuRecord(uuid, key, weights.get(key, 0.0) if model else 0.0, old.verified_at if keep else 0.0,
                            old.last_result if keep else "", node.name, old.code if keep else codes.PENDING, int(g.get("index", 0)),
                            old.verified_since if keep else 0.0, old.fee_covered if old else not self.config.fee_address)
            if not model:
                gpu.code, gpu.last_result = codes.UNKNOWN_MODEL, f"{g.get('name')} with {g.get('vram_mib')} MiB"
            elif g.get("rented"):
                gpu.code = codes.RENTED
            reported[uuid] = gpu
        if set(reported) != set(mine) and mine:
            log.info("%s/%s: GPU set changed, verification starts over", record.hotkey, node.name)
            for gpu in reported.values():
                gpu.verified_at = gpu.verified_since = 0.0
                if gpu.model:
                    gpu.code = codes.GPU_SET_CHANGED
        for uuid in mine:
            record.gpus.pop(uuid, None)
        record.gpus.update(reported)
        try:
            node.price = float(inventory.get("price_usd_per_gpu_hour"))
        except (TypeError, ValueError):
            node.price = None
        try:
            node.disk_total_gib = float(inventory.get("disk_total_gib") or 0)
        except (TypeError, ValueError):
            node.disk_total_gib = 0.0
        node.container_runtime = str(inventory.get("container_runtime") or "")
        address = str(inventory.get("payout_address") or "")
        if address.isalnum() and 20 <= len(address) <= 100:
            record.payout_address = address

    def _due(self, record: MinerRecord, gpu: dict) -> bool:
        g = record.gpus.get(str(gpu.get("uuid")))
        if not g or not g.model or gpu.get("rented"):
            return False  # rented GPUs keep their last pass; deliveries are measured instead
        age = time.time() - g.verified_at
        jitter = random.uniform(0.75, 1.25)
        return age > min(VERIFIED_TTL * 0.9, self.config.recheck_after * jitter)

    async def _challenge(self, client: NodeClient, record: MinerRecord, gpu: dict) -> None:
        g = record.gpus[str(gpu["uuid"])]
        model = classify(str(gpu["name"]), int(gpu["vram_mib"]))
        # The size varies per challenge so nodes cannot tune for one fixed shape.
        n = self.config.challenge_n or random.SystemRandom().choice(self.config.challenge_sizes)
        spec = challenge.Spec(secrets.token_hex(8), secrets.randbits(62), n,
                              min(self.config.max_pairs, challenge.size_for(int(gpu["vram_mib"]), n, self.config.fill)))
        limit = time_limit(spec.flops, spec.elements, model, self.config.timing_slack, self.config.timing_overhead_s)
        try:
            commit = await client.call("POST", "/v1/challenge", {"spec": spec.to_json(), "gpu": gpu["index"]},
                                       timeout=(limit if self.config.timing else 600) + 30)
            if commit.status == 409:
                reason = str(commit.data.get("error", "busy"))
                g.code, g.last_result = (codes.RENTED if reason == "rented" else codes.BUSY), reason
                return
            if commit.status != 200 or commit.data.get("challenge_id") != spec.challenge_id:
                g.code, g.last_result = codes.CHALLENGE_REFUSED, f"HTTP {commit.status}"
                return
            if self.config.timing and commit.seconds > limit:
                g.code, g.last_result = codes.CHALLENGE_SLOW, f"{commit.seconds:.1f}s, limit {limit:.1f}s for {model.key}"
                return
            picks = challenge.pick_rows(spec, self.config.picks_per_gpu, random.SystemRandom())
            opened = await client.call("POST", "/v1/challenge/open", {"challenge_id": spec.challenge_id, "picks": picks}, timeout=120)
            if opened.status != 200:
                g.code, g.last_result = codes.CHALLENGE_REFUSED, f"reveal HTTP {opened.status}"
                return
            fps = np.frombuffer(base64.b64decode(opened.data["fingerprints"]), dtype="<i8")
            verdict = await asyncio.to_thread(challenge.verify, spec, str(commit.data["root"]), fps.tolist(), picks, opened.data["rows"])
        except NodeError as error:
            g.code, g.last_result = self._error_code(error), str(error)
            return
        except (KeyError, TypeError, ValueError) as error:
            verdict = challenge.Verdict(False, f"malformed reveal: {error}")
        if verdict.ok:
            now = time.time()
            if not g.verified_since or now - g.verified_at >= VERIFIED_TTL:
                g.verified_since = now  # first pass, or a gap: the waiting period starts again
            g.verified_at = now
            g.code, g.last_result = codes.OK, f"passed in {commit.seconds:.1f}s"
        else:
            # A wrong answer is not a network accident: zero the miner for a day.
            g.code, g.last_result = codes.CHALLENGE_WRONG, verdict.reason
            record.struck_until = time.time() + STRIKE_SECONDS
            for other in record.gpus.values():
                other.verified_since = 0.0
            log.warning("%s: wrong challenge answer on %s: %s", record.hotkey, g.uuid, verdict.reason)

    def _check_duplicates(self) -> None:
        """A GPU UUID counts once: for the hotkey and node that showed it first."""
        seen: dict[str, str] = {}
        for record in sorted(self.miners.values(), key=lambda m: m.uid):
            for uuid, g in sorted(record.gpus.items(), key=lambda kv: kv[1].node):
                owner = f"{record.hotkey}/{g.node}"
                if uuid in seen and seen[uuid] != owner:
                    g.model, g.weight, g.code, g.last_result = None, 0.0, codes.DUPLICATE_GPU, f"also on {seen[uuid]}"
                else:
                    seen[uuid] = owner

    def _write_sellable(self, snapshot: Snapshot) -> None:
        """Only GPUs that passed a challenge recently, on reachable nodes of
        miners in good standing. One entry per node."""
        path = self.config.sellable_file
        if not path:
            return
        now = time.time()
        listed = {n.hotkey for n in snapshot.neurons if n.axon}
        caps = self._caps()
        nodes = []
        for record in self.miners.values():
            if now < record.struck_until or record.hotkey not in listed or not record.fee_paid:
                continue
            for node in record.nodes.values():
                if price_gate(node, record, caps) != codes.OK:
                    continue
                verified = [g.uuid for g in record.gpus.values()
                            if g.node == node.name and g.model and g.fee_covered and now - g.verified_at < VERIFIED_TTL]
                if verified and node.availability >= 0.5 and node.code in (codes.OK,):
                    nodes.append({"hotkey": record.hotkey, "node": node.name, "url": node.url, "verified_gpus": verified,
                                  "availability": round(node.availability, 3), "payout_address": record.payout_address})
        self._write(path, {"generated_at": int(now), "block": snapshot.block, "nodes": nodes})

    def _write_status(self, snapshot: Snapshot) -> None:
        """Everything a miner needs to see why its machines are or are not sold."""
        path = self.config.status_file
        if not path:
            return
        now = time.time()
        caps = self._caps()
        policy = self.config.policy
        coldkey_of = {n.hotkey: n.coldkey for n in snapshot.neurons}
        fees = self._per_model("fees")
        miners = {}
        for record in self.miners.values():
            struck = now < record.struck_until
            miners[record.hotkey] = {
                "uid": record.uid,
                "struck_until": int(record.struck_until) if struck else None,
                "activated": record.fee_paid,
                "activation_fees_tao": fees,
                "activation_credit_tao": round(max(0, self.credit_left.get(coldkey_of.get(record.hotkey, ""), 0)) / RAO_PER_TAO, 4),
                "payout_address": record.payout_address,
                "nodes": [{
                    "name": n.name, "url": n.url, "code": codes.STRUCK if struck else n.code, "detail": n.detail,
                    "idle_pay": idle_gate(n, record, policy, caps),
                    "price_cap": caps.get(_model_of(record, n.name) or ""),
                    "availability": round(n.availability, 3),
                    "gpus": [{"index": g.index, "uuid": g.uuid, "model": g.model,
                              "code": _fee_code(g, _gpu_code(g, now, policy)),
                              "detail": g.last_result, "verified_at": int(g.verified_at) or None,
                              "idle_pay_from": int(g.verified_since + policy.probation_seconds) if g.verified_since else None}
                             for g in sorted(record.gpus.values(), key=lambda g: g.index) if g.node == n.name],
                } for n in record.nodes.values()],
            }
        self._write(path, {"generated_at": int(now), "block": snapshot.block, "activation_fees_tao": fees, "miners": miners})

    async def _set_weights(self, snapshot: Snapshot, now: float) -> dict[int, float]:
        weights, detail = compute_weights(list(self.miners.values()), self._rentals(), self._reserve_uid(snapshot), self.config.target_capacity, now,
                                          self.config.policy, self._caps())
        try:
            await asyncio.to_thread(self.chain.set_weights, weights)
        except Exception as error:
            # Usually the chain's rate limit (also right after registering).
            # last_weights stays put, so the next round tries again.
            log.warning("weights not set yet: %s", error)
            return None
        self.last_weights = now
        self._save()
        if self.config.summary_file:
            summary = {"block": snapshot.block, "validator": self.keypair.ss58_address, "weights": weights, "miners": detail}
            self.config.summary_file.write_text(json.dumps(summary, indent=1))
        log.info("weights set for %d UIDs (reserve share %.3f)", len(weights), weights.get(self._reserve_uid(snapshot), 0.0))
        return weights

    async def run_forever(self, round_seconds: float = 300.0) -> None:
        while True:
            try:
                await self.run_round()
            except Exception:
                log.exception("round failed")
            await asyncio.sleep(round_seconds)


def _model_of(record: MinerRecord, node: str) -> str | None:
    from .scoring import _node_model

    return _node_model(record, node)


def _fee_code(g: GpuRecord, code: str) -> str:
    """A GPU that passed but has no activation fee yet shows NOT_ACTIVATED."""
    return codes.NOT_ACTIVATED if not g.fee_covered and code in (codes.OK, codes.PROBATION) else code


def _gpu_code(g: GpuRecord, now: float, policy: Policy) -> str:
    if g.code != codes.OK or not g.model:
        return g.code
    if now - g.verified_at >= VERIFIED_TTL:
        return codes.PENDING  # its pass expired; checked again soon
    return codes.PROBATION if in_probation(g, now, policy) else codes.OK
