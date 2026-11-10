"""weir-node: the agent a miner runs next to its GPUs.

Every request must be signed (btauth/1) by an allowed hotkey and addressed to
this node's hotkey. Validators may read the inventory and run challenges; only
the Weir platform key may place, restart or release orders. Every response is
signed with the node's hotkey.

The node holds the miner's hotkey only. A coldkey never belongs on this box.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import bittensor as bt
import numpy as np
from aiohttp import web
from bittensor_core import Keypair

from .. import challenge
from ..protocol import AGENT_VERSION, NONCE_HEADER, RESPONSE_HEADER, sign_response
from . import gpus as gpu_mod
from .runtime import Runtime, RuntimeFailure, StateOnlyRuntime

CHALLENGE_TTL = 600  # seconds a commitment stays open
MAX_OPEN_CHALLENGES = 64
MAX_BODY = 1 << 20

log = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    hotkey: Keypair
    validators: set[str]  # hotkeys allowed to read and challenge
    platform: set[str]  # hotkeys allowed to manage orders
    gpus: list[gpu_mod.Gpu]
    simulated: bool = False
    price_usd_per_gpu_hour: str = "0"
    state_dir: Path = Path("/var/lib/weir-node")
    public_host: str = ""
    solver: Callable | None = None  # tests only; defaults to gpus.solve
    runtime: Runtime = field(default_factory=StateOnlyRuntime)
    ssh_ports: tuple[int, int] = (42000, 42999)  # host ports handed to tenants
    images: list[str] = field(default_factory=list)  # Weir image ids tenants can get
    cpus_per_gpu: float = 8.0
    memory_gib_per_gpu: int = 48
    payout_address: str = ""  # stablecoin address for rent payouts
    name: str = "main"  # this machine's name within the miner's fleet
    fleet: list[tuple[str, str]] = field(default_factory=list)  # (name, url) of all machines; set on the node at the chain address


@dataclass
class OpenChallenge:
    spec: challenge.Spec
    gpu: int
    fingerprints: np.ndarray
    expires: float


@dataclass
class Order:
    """An order on this node, run by the configured runtime."""

    order_id: str
    gpus: list[int]
    image: str
    hours: int
    ssh_keys: list[str]
    state: str = "starting"  # starting -> running -> released; or failed
    created: float = field(default_factory=time.time)
    port: int = 0
    error: str = ""

    def view(self, host: str) -> dict:
        connection = {"host": host, "port": self.port, "user": "root"} if self.state == "running" else None
        view = {"order_id": self.order_id, "state": self.state, "gpus": self.gpus, "connection": connection}
        if self.error:
            view["error"] = self.error
        return view


class Node:
    def __init__(self, config: NodeConfig):
        self.config = config
        self.nonces = bt.http_auth.InMemoryNonceStore()
        self.locks = {g.index: asyncio.Lock() for g in config.gpus}
        self.open: dict[tuple[str, str], OpenChallenge] = {}
        self.orders: dict[str, Order] = {}
        self.orders_file = config.state_dir / "orders.json"
        self.tasks: set[asyncio.Task] = set()
        self.launching: set[str] = set()  # order ids with a launch in flight
        self.stats = {"challenges_total": 0, "challenge_seconds_last": 0.0, "requests_refused_total": 0}
        self._load_orders()

    # -- persistence -------------------------------------------------------

    def _load_orders(self) -> None:
        if self.orders_file.exists():
            for raw in json.loads(self.orders_file.read_text()):
                order = Order(**raw)
                self.orders[order.order_id] = order

    def _save_orders(self) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.orders_file.with_suffix(".tmp")
        tmp.write_text(json.dumps([vars(o) for o in self.orders.values()]))
        os.replace(tmp, self.orders_file)

    def active(self) -> list[Order]:
        return [o for o in self.orders.values() if o.state not in ("released", "failed")]

    def rented(self) -> set[int]:
        return {i for o in self.active() for i in o.gpus}

    def _free_port(self) -> int | None:
        used = {o.port for o in self.active()}
        low, high = self.config.ssh_ports
        free = [p for p in range(low, high + 1) if p not in used]
        return free[secrets.randbelow(len(free))] if free else None

    def _start(self, order: Order) -> None:
        async def run():
            self.launching.add(order.order_id)
            try:
                await self.config.runtime.launch(order)
            except RuntimeFailure as error:
                order.state, order.error = "failed", str(error)
                self._save_orders()
            finally:
                self.launching.discard(order.order_id)
        task = asyncio.get_running_loop().create_task(run())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def resume(self, app: web.Application) -> None:
        """After a node restart, finish launches that were in flight."""
        for order in self.orders.values():
            if order.state == "starting":
                self._start(order)

    # -- auth --------------------------------------------------------------

    async def authenticate(self, request: web.Request, allowed: set[str]) -> tuple[str, bytes]:
        body = await request.read()
        if len(body) > MAX_BODY:
            raise web.HTTPRequestEntityTooLarge(max_size=MAX_BODY, actual_size=len(body))
        try:
            caller = bt.http_auth.verify(
                request.headers, body,
                method=request.method, path=request.raw_path,
                self_hotkey_ss58=self.config.hotkey.ss58_address,
                max_age=30.0, nonce_store=self.nonces,
            )
        except bt.http_auth.AuthError as error:
            self.stats["requests_refused_total"] += 1
            raise web.HTTPUnauthorized(text=str(error))
        if caller.hotkey_ss58 not in allowed:
            self.stats["requests_refused_total"] += 1
            raise web.HTTPForbidden(text="caller is not allowed")
        return caller.hotkey_ss58, body

    def reply(self, request: web.Request, data: dict, status: int = 200) -> web.Response:
        body = json.dumps(data, separators=(",", ":")).encode()
        signature = sign_response(self.config.hotkey, request.headers.get(NONCE_HEADER, ""), body)
        return web.Response(body=body, status=status, content_type="application/json", headers={RESPONSE_HEADER: signature})

    # -- validator API -----------------------------------------------------

    async def inventory(self, request: web.Request) -> web.Response:
        await self.authenticate(request, self.config.validators | self.config.platform)
        rented = self.rented()
        return self.reply(request, {
            "agent_version": AGENT_VERSION,
            "name": self.config.name,
            "hotkey": self.config.hotkey.ss58_address,
            "gpus": [dict(gpu_mod.as_json([g])[0], rented=g.index in rented) for g in self.config.gpus],
            "price_usd_per_gpu_hour": self.config.price_usd_per_gpu_hour,
            "cpus_per_gpu": self.config.cpus_per_gpu,
            "memory_gib_per_gpu": self.config.memory_gib_per_gpu,
            "images": self.config.images,
            "payout_address": self.config.payout_address,
            "container_runtime": getattr(self.config.runtime, "docker_runtime", "") or "",
            **gpu_mod.host_facts(),
        })

    async def start_challenge(self, request: web.Request) -> web.Response:
        caller, body = await self.authenticate(request, self.config.validators)
        try:
            data = json.loads(body)
            spec = challenge.Spec.from_json(data["spec"])
            gpu = next(g for g in self.config.gpus if g.index == int(data["gpu"]))
        except (ValueError, KeyError, StopIteration, TypeError):
            return self.reply(request, {"error": "bad challenge"}, 400)
        if gpu.index in self.rented():
            return self.reply(request, {"error": "rented"}, 409)
        lock = self.locks[gpu.index]
        if lock.locked():
            return self.reply(request, {"error": "busy"}, 409)
        async with lock:
            started = time.perf_counter()
            fps = await asyncio.get_running_loop().run_in_executor(None, self.config.solver or gpu_mod.solve, spec, gpu, self.config.simulated)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.stats["challenges_total"] += 1
        self.stats["challenge_seconds_last"] = elapsed_ms / 1000
        self._expire()
        if len(self.open) >= MAX_OPEN_CHALLENGES:
            self.open.pop(next(iter(self.open)))
        self.open[(caller, spec.challenge_id)] = OpenChallenge(spec, gpu.index, fps, time.time() + CHALLENGE_TTL)
        return self.reply(request, {"challenge_id": spec.challenge_id, "gpu": gpu.index, "root": challenge.commitment(spec, fps), "elapsed_ms": elapsed_ms})

    async def open_challenge(self, request: web.Request) -> web.Response:
        caller, body = await self.authenticate(request, self.config.validators)
        try:
            data = json.loads(body)
            entry = self.open.pop((caller, str(data["challenge_id"])))
            picks = [(int(b), int(i)) for b, i in data["picks"]][:16]
        except (ValueError, KeyError, TypeError):
            return self.reply(request, {"error": "unknown challenge"}, 404)
        if entry.expires < time.time():
            return self.reply(request, {"error": "expired"}, 410)
        rows = await asyncio.get_running_loop().run_in_executor(None, challenge.reveal_rows, entry.spec, picks)
        fps = base64.b64encode(np.ascontiguousarray(entry.fingerprints, dtype="<i8").tobytes()).decode()
        return self.reply(request, {"challenge_id": entry.spec.challenge_id, "fingerprints": fps, "rows": rows})

    async def fleet(self, request: web.Request) -> web.Response:
        """The miner's machines. Each one signs with the same hotkey, so a
        listed machine that is not the miner's fails verification."""
        await self.authenticate(request, self.config.validators | self.config.platform)
        if not self.config.fleet:
            return self.reply(request, {"error": "single machine"}, 404)
        return self.reply(request, {"nodes": [{"name": n, "url": u} for n, u in self.config.fleet]})

    def _expire(self) -> None:
        now = time.time()
        for key in [k for k, v in self.open.items() if v.expires < now]:
            del self.open[key]

    # -- platform API (orders) --------------------------------------------

    async def place_order(self, request: web.Request) -> web.Response:
        _, body = await self.authenticate(request, self.config.platform)
        try:
            data = json.loads(body)
            order_id, count = str(data["order_id"]), int(data["gpus"])
        except (ValueError, KeyError, TypeError):
            return self.reply(request, {"error": "bad order"}, 400)
        if order_id in self.orders:  # idempotent: same id, same answer
            return self.reply(request, self.orders[order_id].view(self.config.public_host))
        free = [g.index for g in self.config.gpus if g.index not in self.rented()]
        if count < 1 or count > len(free):
            return self.reply(request, {"error": "not enough free gpus"}, 409)
        port = self._free_port()
        keys = [str(k).strip() for k in data.get("ssh_keys", []) if str(k).strip()]
        if port is None or not keys:
            return self.reply(request, {"error": "no free port" if keys else "no ssh keys"}, 409 if keys else 400)
        order = Order(order_id, free[:count], str(data.get("image", "")), int(data.get("hours", 1)), keys, port=port)
        self.orders[order_id] = order
        try:
            self._save_orders()  # saved before launching, so a crash cannot lose it
        except Exception:
            del self.orders[order_id]  # not taken: the platform sees 404 and refunds
            raise
        self._start(order)
        return self.reply(request, order.view(self.config.public_host), 201)

    async def order_state(self, request: web.Request) -> web.Response:
        await self.authenticate(request, self.config.platform)
        order = self.orders.get(request.match_info["order_id"])
        if not order:
            return self.reply(request, {"error": "unknown order", "state": "gone"}, 404)
        if order.state in ("starting", "running"):
            try:
                seen = await self.config.runtime.status(order)
            except RuntimeFailure:
                seen = order.state  # cannot tell right now; keep what we had
            if seen == "gone" and order.order_id in self.launching:
                seen = "starting"
            new = {"running": "running", "starting": "starting", "failed": "failed", "gone": "failed"}[seen]
            if new != order.state:
                order.state = new
                if new == "failed":
                    order.error = order.error or "the machine stopped"
                self._save_orders()
        return self.reply(request, order.view(self.config.public_host))

    async def order_action(self, request: web.Request) -> web.Response:
        await self.authenticate(request, self.config.platform)
        order = self.orders.get(request.match_info["order_id"])
        action = request.match_info["action"]
        if not order:
            return self.reply(request, {"error": "unknown order", "state": "gone"}, 404)
        try:
            if action == "release":
                await self.config.runtime.release(order)  # releasing twice is fine
                order.state = "released"
            elif action == "restart":
                if order.state in ("released", "failed"):
                    return self.reply(request, {"error": order.state}, 409)
                await self.config.runtime.restart(order)
                order.state = "starting"
            else:
                return self.reply(request, {"error": "unknown action"}, 404)
        except RuntimeFailure as error:
            return self.reply(request, {"error": str(error)}, 502)
        self._save_orders()
        return self.reply(request, order.view(self.config.public_host))

    def metrics_text(self) -> str:
        """Prometheus text format, for the miner's own monitoring."""
        rented = self.rented()
        lines = [
            "# TYPE weir_gpus gauge", f"weir_gpus {len(self.config.gpus)}",
            "# TYPE weir_gpus_rented gauge", f"weir_gpus_rented {len(rented)}",
            "# TYPE weir_orders_active gauge", f"weir_orders_active {len(self.active())}",
            "# TYPE weir_challenges_total counter", f"weir_challenges_total {self.stats['challenges_total']}",
            "# TYPE weir_challenge_seconds_last gauge", f"weir_challenge_seconds_last {self.stats['challenge_seconds_last']}",
            "# TYPE weir_requests_refused_total counter", f"weir_requests_refused_total {self.stats['requests_refused_total']}",
        ]
        lines += [f'weir_gpu_rented{{index="{g.index}",name="{g.name}"}} {int(g.index in rented)}' for g in self.config.gpus]
        return "\n".join(lines) + "\n"

    def metrics_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([web.get("/metrics", lambda request: web.Response(text=self.metrics_text(), content_type="text/plain"))])
        return app

    @web.middleware
    async def sign_errors(self, request: web.Request, handler):
        """Refusals (401, 403, 404, 413...) and unexpected errors are signed like
        every other answer, so the caller can trust and act on them."""
        try:
            return await handler(request)
        except web.HTTPException as error:
            if error.status < 400:
                raise
            return self.reply(request, {"error": error.text or error.reason}, error.status)
        except Exception as error:  # noqa: BLE001 - a signed 500 lets the platform ask what happened
            log.exception("%s %s failed", request.method, request.path)
            return self.reply(request, {"error": f"node error: {type(error).__name__}: {error}"[:300]}, 500)

    def app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_BODY, middlewares=[self.sign_errors])
        app.on_startup.append(self.resume)
        app.add_routes([
            web.get("/v1/inventory", self.inventory),
            web.get("/v1/fleet", self.fleet),
            web.post("/v1/challenge", self.start_challenge),
            web.post("/v1/challenge/open", self.open_challenge),
            web.post("/v1/orders", self.place_order),
            web.get("/v1/orders/{order_id}", self.order_state),
            web.post("/v1/orders/{order_id}/{action}", self.order_action),
        ])
        return app
