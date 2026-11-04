"""A miner with several machines under one hotkey, and splitting one machine."""

import json

import aiohttp
import pytest
from aiohttp.test_utils import TestServer
from bittensor_core import Keypair

from weir_subnet.node import gpus
from weir_subnet.node.server import Node, NodeConfig
from weir_subnet.validator.chain import Neuron, Snapshot
from weir_subnet.validator.client import NodeClient
from weir_subnet.validator.scoring import Policy
from weir_subnet.validator.validator import Validator, ValidatorConfig


def _platform(tmp_path):
    import json
    import time

    path = tmp_path / "platform.json"
    path.write_text(json.dumps({"generated_at": time.time(), "caps": {"rtx4090-24g": 1.0, "a100-80g": 1.0, "h100-80g": 1.4}}))
    return path

VALIDATOR, PLATFORM, OWNER = (Keypair.create_from_uri(f"//{n}") for n in ("FleetValidator", "FleetPlatform", "FleetOwner"))
MINER = Keypair.create_from_uri("//FleetMiner")


class Chain:
    def __init__(self, neurons):
        self.neurons = neurons

    def snapshot(self):
        return Snapshot(5, OWNER.ss58_address, self.neurons)

    def set_weights(self, weights):
        self.weights = weights


async def node(tmp_path, name, gpu_spec, sign_with=MINER, fleet=None, calls_itself=None, keep=None):
    config = NodeConfig(hotkey=sign_with, validators={VALIDATOR.ss58_address}, platform={PLATFORM.ss58_address},
                        gpus=gpus.simulated(gpu_spec, f"{name}-box"), simulated=True, price_usd_per_gpu_hour="0.30",
                        state_dir=tmp_path / name, name=calls_itself or name, fleet=fleet or [])
    agent = Node(config)
    if keep is not None:
        keep.append(agent)
    server = TestServer(agent.app(), host="127.0.0.1")
    await server.start_server()
    return server, config


async def test_fleet_of_machines_under_one_hotkey(tmp_path):
    # big: 8 GPUs; small: 2; stolen: someone else's machine listed by this miner;
    # renamed: a machine whose own name does not match the fleet entry.
    big, big_cfg = await node(tmp_path, "big", "RTX 4090:24564:8")
    small, _ = await node(tmp_path, "small", "RTX 4090:24564:2")
    stolen, _ = await node(tmp_path, "stolen", "RTX 4090:24564:4", sign_with=Keypair.create_from_uri("//SomeoneElse"))
    renamed, _ = await node(tmp_path, "renamed", "RTX 4090:24564:1", calls_itself="other")
    url = lambda s: f"http://127.0.0.1:{s.port}"  # noqa: E731
    big_cfg.fleet = [("big", url(big)), ("small", url(small)), ("stolen", url(stolen)), ("renamed", url(renamed))]
    try:
        chain = Chain([Neuron(0, OWNER.ss58_address, None, True), Neuron(1, MINER.ss58_address, f"127.0.0.1:{big.port}", False)])
        config = ValidatorConfig(state_file=tmp_path / "v.json", sellable_file=tmp_path / "sellable.json", status_file=tmp_path / "status.json",
                                 challenge_n=128, max_pairs=1, timing=False, weights_every=0, target_capacity=32,
                                 policy=Policy(require_sysbox=False, min_disk_ratio=0, probation_seconds=0), platform_file=_platform(tmp_path))
        weights = await Validator(VALIDATOR, chain, config).run_round()
        assert weights[1] == pytest.approx(10 * 0.80 / 32)  # 8 + 2 idle verified GPUs count; the others do not

        sellable = {n["node"]: n for n in json.loads((tmp_path / "sellable.json").read_text())["nodes"]}
        assert set(sellable) == {"big", "small"}
        assert len(sellable["big"]["verified_gpus"]) == 8 and sellable["small"]["url"] == url(small)
        assert all(n["hotkey"] == MINER.ss58_address for n in sellable.values())

        status = {n["name"]: n for n in json.loads((tmp_path / "status.json").read_text())["miners"][MINER.ss58_address]["nodes"]}
        assert status["big"]["code"] == "OK" and all(g["code"] == "OK" for g in status["big"]["gpus"])
        assert status["stolen"]["code"] == "BAD_SIGNATURE"
        assert status["renamed"]["code"] == "FLEET_INVALID"

        # A restart keeps the per-machine history.
        again = Validator(VALIDATOR, chain, config)
        assert set(again.miners[MINER.ss58_address].nodes) == {"big", "small", "stolen", "renamed"}
    finally:
        for s in (big, small, stolen, renamed):
            await s.close()


async def test_one_machine_split_between_orders(tmp_path):
    agents = []
    server, _ = await node(tmp_path, "split", "RTX 4090:24564:8", keep=agents)
    try:
        async with aiohttp.ClientSession() as session:
            platform = NodeClient(session, PLATFORM, f"http://127.0.0.1:{server.port}", MINER.ss58_address)
            keys = ["ssh-ed25519 AAAA tenant"]
            sizes = {"a": 4, "b": 2, "c": 1}
            placed = {}
            for order_id, count in sizes.items():
                reply = await platform.call("POST", "/v1/orders", {"order_id": order_id, "gpus": count, "ssh_keys": keys})
                assert reply.status == 201, reply.data
                placed[order_id] = reply.data["gpus"]
            # Each order has its own GPUs; no GPU is given twice.
            used = [i for g in placed.values() for i in g]
            assert len(used) == len(set(used)) == 7
            inventory = (await platform.call("GET", "/v1/inventory")).data
            assert sum(g["rented"] for g in inventory["gpus"]) == 7
            # Only 1 GPU left: a 2-GPU order does not fit, a 1-GPU order does.
            assert (await platform.call("POST", "/v1/orders", {"order_id": "d", "gpus": 2, "ssh_keys": keys})).status == 409
            assert (await platform.call("POST", "/v1/orders", {"order_id": "e", "gpus": 1, "ssh_keys": keys})).status == 201
            # Metrics for the miner's own monitoring.
            metrics = agents[0].metrics_text()
            assert "weir_gpus 8" in metrics and "weir_gpus_rented 8" in metrics and "weir_orders_active 4" in metrics
            # Releasing one order frees exactly its GPUs.
            await platform.call("POST", "/v1/orders/a/release")
            inventory = (await platform.call("GET", "/v1/inventory")).data
            free = sorted(g["index"] for g in inventory["gpus"] if not g["rented"])
            assert free == sorted(placed["a"])
    finally:
        await server.close()
