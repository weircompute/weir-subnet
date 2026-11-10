"""Validator against real node servers on localhost, with a fake chain."""

import json
import time

import aiohttp
import numpy as np
import pytest
from aiohttp.test_utils import TestServer
from bittensor_core import Keypair

from weir_subnet import challenge
from weir_subnet.node import gpus
from weir_subnet.node.server import Node, NodeConfig
from weir_subnet.validator.chain import Neuron, Snapshot
from weir_subnet.validator.client import NodeClient, NodeError
from weir_subnet.validator.scoring import Policy
from weir_subnet.validator.validator import Validator, ValidatorConfig


def key(name):
    return Keypair.create_from_uri(f"//{name}")


VALIDATOR, PLATFORM, OWNER = key("WeirValidator"), key("WeirPlatform"), key("WeirOwner")
NO_WAIT = Policy(require_sysbox=False, min_disk_ratio=0, probation_seconds=0)


def platform_file(tmp_path, caps=None, generated_at=None, activation_credit=None, **extra):
    path = tmp_path / "platform.json"
    path.write_text(json.dumps({"generated_at": generated_at or time.time(), "caps": caps if caps is not None else {"rtx4090-24g": 1.0},
                                "activation_credit": activation_credit or {},
                                "fees": {"h100-80g": 10.0, "rtx4090-24g": 3.7}, **extra}))
    return path


def cheating_solver(spec, gpu, simulated):
    """Does the first matrix, guesses the rest."""
    fps = challenge.solve_numpy(challenge.Spec(spec.challenge_id, spec.seed, spec.n, 1))
    rest = np.random.randint(-(2**62), 2**62, size=(spec.count - 1, spec.n), dtype=np.int64)
    return np.concatenate([fps, rest])


class FakeChain:
    def __init__(self, neurons, transfers=()):
        self.neurons = neurons
        self.weights = []
        self.transfers = list(transfers)  # (to, from, rao, block)
        self.scanned = []

    def snapshot(self):
        return Snapshot(100, OWNER.ss58_address, self.neurons)

    def set_weights(self, weights):
        self.weights.append(weights)

    def transfers_to(self, address, start, end):
        self.scanned.append((start, end))
        return [(sender, rao, block) for to, sender, rao, block in self.transfers if to == address and start <= block <= end]


async def start_node(tmp_path, name, hotkey, gpu_spec="RTX 4090:24564:2", price="0.60", solver=None, sign_with=None, uuid_salt=None):
    config = NodeConfig(
        hotkey=sign_with or hotkey, validators={VALIDATOR.ss58_address}, platform={PLATFORM.ss58_address},
        gpus=gpus.simulated(gpu_spec, uuid_salt or name), simulated=True, price_usd_per_gpu_hour=price,
        state_dir=tmp_path / name, public_host="203.0.113.9", solver=solver,
        payout_address="TQn9Y2khEsLJW1ChVWFMSMeRDow5KcbLSE",
    )
    server = TestServer(Node(config).app(), host="127.0.0.1")
    await server.start_server()
    return server


@pytest.fixture
async def network(tmp_path):
    honest_key, cheat_key, imposter_key, offline_key = key("Honest"), key("Cheat"), key("Imposter"), key("Offline")
    honest = await start_node(tmp_path, "honest", honest_key)
    cheat = await start_node(tmp_path, "cheat", cheat_key, solver=cheating_solver)
    # Listed under one hotkey, but the server signs with another.
    imposter = await start_node(tmp_path, "imposter", imposter_key, sign_with=key("SomeoneElse"))
    neurons = [
        Neuron(0, OWNER.ss58_address, None, True),
        Neuron(1, honest_key.ss58_address, f"127.0.0.1:{honest.port}", False),
        Neuron(2, cheat_key.ss58_address, f"127.0.0.1:{cheat.port}", False),
        Neuron(3, imposter_key.ss58_address, f"127.0.0.1:{imposter.port}", False),
        Neuron(4, offline_key.ss58_address, "127.0.0.1:9", False),
        Neuron(5, key("NoAxon").ss58_address, None, False),
    ]
    chain = FakeChain(neurons)
    config = ValidatorConfig(state_file=tmp_path / "v" / "state.json", summary_file=tmp_path / "summary.json", sellable_file=tmp_path / "sellable.json",
                             challenge_n=128, max_pairs=3, timing=False, weights_every=0, target_capacity=32, policy=NO_WAIT,
                             platform_file=platform_file(tmp_path))
    yield chain, Validator(VALIDATOR, chain, config), {"honest": honest, "honest_key": honest_key, "cheat_key": cheat_key}
    for s in (honest, cheat, imposter):
        await s.close()


async def test_round_scores_only_the_honest_miner(network, tmp_path):
    chain, validator, nodes = network
    weights = await validator.run_round()
    assert weights is not None and chain.weights == [weights]
    assert set(weights) == {0, 1}  # owner (no reserve set) and the honest miner
    assert weights[1] == pytest.approx(2 * 0.80 / 32)  # 2 idle verified RTX 4090s of a 32-unit target
    assert weights[0] == pytest.approx(1 - 2 * 0.80 / 32)

    by_key = validator.miners
    honest = by_key[nodes["honest_key"].ss58_address]
    assert all(g.code == "OK" for g in honest.gpus.values())
    cheat = by_key[nodes["cheat_key"].ss58_address]
    assert cheat.struck_until > 0
    assert any(g.code == "CHALLENGE_WRONG" for g in cheat.gpus.values())
    imposter = by_key[key("Imposter").ss58_address]
    assert imposter.availability == 0 and "signed" in imposter.note
    assert by_key[key("Offline").ss58_address].availability == 0

    sellable = json.loads((tmp_path / "sellable.json").read_text())["nodes"]
    assert [n["hotkey"] for n in sellable] == [nodes["honest_key"].ss58_address]
    assert len(sellable[0]["verified_gpus"]) == 2 and sellable[0]["url"] == f"http://127.0.0.1:{nodes['honest'].port}"
    assert sellable[0]["payout_address"] == "TQn9Y2khEsLJW1ChVWFMSMeRDow5KcbLSE"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["miners"][nodes["honest_key"].ss58_address]["idle_gpus"] == 2


async def test_state_survives_a_restart(network):
    chain, validator, nodes = network
    await validator.run_round()
    again = Validator(VALIDATOR, chain, validator.config)
    honest = again.miners[nodes["honest_key"].ss58_address]
    assert all(g.verified_at > 0 for g in honest.gpus.values())
    assert again.miners[nodes["cheat_key"].ss58_address].struck_until > 0


async def test_same_gpu_under_two_hotkeys_counts_once(tmp_path):
    a, b = key("TwinA"), key("TwinB")
    na = await start_node(tmp_path, "twin-a", a, uuid_salt="same-box")
    nb = await start_node(tmp_path, "twin-b", b, uuid_salt="same-box")  # reports the same GPU UUIDs
    try:
        chain = FakeChain([
            Neuron(0, OWNER.ss58_address, None, True),
            Neuron(1, a.ss58_address, f"127.0.0.1:{na.port}", False),
            Neuron(2, b.ss58_address, f"127.0.0.1:{nb.port}", False),
        ])
        validator = Validator(VALIDATOR, chain, ValidatorConfig(state_file=tmp_path / "s.json", challenge_n=128, max_pairs=2,
                                                                 timing=False, weights_every=0, policy=NO_WAIT, platform_file=platform_file(tmp_path)))
        weights = await validator.run_round()
        assert 1 in weights and 2 not in weights
        assert all(g.code == "DUPLICATE_GPU" for g in validator.miners[b.ss58_address].gpus.values())
    finally:
        await na.close()
        await nb.close()


async def test_node_rejects_strangers_and_misaddressed_requests(network):
    _, _, nodes = network
    server, hotkey = nodes["honest"], nodes["honest_key"]
    async with aiohttp.ClientSession() as session:
        base = f"http://127.0.0.1:{server.port}"
        stranger = NodeClient(session, key("Stranger"), base, hotkey.ss58_address)
        assert (await stranger.call("GET", "/v1/inventory")).status == 403
        misaddressed = NodeClient(session, VALIDATOR, base, key("Elsewhere").ss58_address)
        with pytest.raises(NodeError):  # 401 is signed by the real node, not by "Elsewhere"
            await misaddressed.call("GET", "/v1/inventory")
        # Validators may not touch orders.
        assert (await NodeClient(session, VALIDATOR, base, hotkey.ss58_address).call("POST", "/v1/orders", {"order_id": "o1", "gpus": 1})).status == 403


async def test_orders_are_idempotent_and_block_challenges(network):
    _, _, nodes = network
    server, hotkey = nodes["honest"], nodes["honest_key"]
    async with aiohttp.ClientSession() as session:
        base = f"http://127.0.0.1:{server.port}"
        platform = NodeClient(session, PLATFORM, base, hotkey.ss58_address)
        first = await platform.call("POST", "/v1/orders", {"order_id": "wr-1", "gpus": 1, "hours": 2, "ssh_keys": ["ssh-ed25519 AAAA test"]})
        again = await platform.call("POST", "/v1/orders", {"order_id": "wr-1", "gpus": 1, "hours": 2, "ssh_keys": ["ssh-ed25519 AAAA test"]})
        assert first.status == 201 and again.status == 200 and first.data["gpus"] == again.data["gpus"]
        state = await platform.call("GET", "/v1/orders/wr-1")
        assert state.data["state"] == "running" and state.data["connection"]["host"] == "203.0.113.9"

        validator = NodeClient(session, VALIDATOR, base, hotkey.ss58_address)
        inventory = (await validator.call("GET", "/v1/inventory")).data
        rented = [g for g in inventory["gpus"] if g["rented"]]
        assert len(rented) == 1
        spec = challenge.Spec("c", 5, 128, 1)
        busy = await validator.call("POST", "/v1/challenge", {"spec": spec.to_json(), "gpu": rented[0]["index"]})
        assert busy.status == 409 and busy.data["error"] == "rented"

        assert (await platform.call("POST", "/v1/orders", {"order_id": "wr-2", "gpus": 2, "ssh_keys": ["ssh-ed25519 AAAA test"]})).status == 409
        assert (await platform.call("POST", "/v1/orders/wr-1/release")).data["state"] == "released"
        assert (await platform.call("POST", "/v1/orders/wr-1/release")).data["state"] == "released"
        assert (await platform.call("POST", "/v1/orders", {"order_id": "wr-2", "gpus": 2, "ssh_keys": ["ssh-ed25519 AAAA test"]})).status == 201


async def test_fees_reserve_and_probation(tmp_path):
    """One paid miner, one unpaid miner, one over the price cap and the reserve, end to end."""
    paid, unpaid, reserve = key("Paid"), key("Unpaid"), key("Reserve")
    fee_address = key("FeeWallet").ss58_address
    np_ = await start_node(tmp_path, "paid", paid)
    nu = await start_node(tmp_path, "unpaid", unpaid)
    pricey = await start_node(tmp_path, "pricey", key("Pricey"), price="1.50")
    try:
        neurons = [
            Neuron(0, OWNER.ss58_address, None, True, "cold-owner"),
            Neuron(1, paid.ss58_address, f"127.0.0.1:{np_.port}", False, "cold-paid"),
            Neuron(2, unpaid.ss58_address, f"127.0.0.1:{nu.port}", False, "cold-unpaid"),
            Neuron(4, reserve.ss58_address, None, False, "cold-reserve"),
            Neuron(5, key("Pricey").ss58_address, f"127.0.0.1:{pricey.port}", False, "cold-pricey"),
        ]
        transfers = [(fee_address, "cold-paid", 10 * 10**9, 95), (fee_address, "cold-unpaid", 3 * 10**9, 96),
                     (fee_address, "cold-pricey", 10 * 10**9, 97), ("elsewhere", "cold-unpaid", 50 * 10**9, 97)]
        chain = FakeChain(neurons, transfers)
        config = ValidatorConfig(state_file=tmp_path / "s.json", summary_file=tmp_path / "summary.json", sellable_file=tmp_path / "sellable.json",
                                 status_file=tmp_path / "status.json", challenge_n=128, max_pairs=2, timing=False, weights_every=0,
                                 target_capacity=32, policy=Policy(require_sysbox=False, min_disk_ratio=0),
                                 platform_file=platform_file(tmp_path),
                                 reserve_hotkey=reserve.ss58_address,
                                 fee_address=fee_address, fee_start_block=90)
        validator = Validator(VALIDATOR, chain, config)
        weights = await validator.run_round()
        unit = 0.80 / 32
        # In probation nobody earns idle pay yet; everything goes to the reserve.
        assert weights == pytest.approx({4: 1.0})
        assert chain.scanned == [(90, 100)]
        status = json.loads((tmp_path / "status.json").read_text())["miners"]
        assert status[paid.ss58_address]["activated"] and not status[unpaid.ss58_address]["activated"]
        assert {g["code"] for g in status[paid.ss58_address]["nodes"][0]["gpus"]} == {"PROBATION"}
        assert status[paid.ss58_address]["nodes"][0]["gpus"][0]["idle_pay_from"] > time.time() + 13 * 86400
        assert status[key("Pricey").ss58_address]["nodes"][0]["idle_pay"] == "PRICE_OVER_CAP"
        sellable = json.loads((tmp_path / "sellable.json").read_text())["nodes"]
        assert [n["hotkey"] for n in sellable] == [paid.ss58_address]  # unpaid and over-cap miners are not sold

        # After the waiting period the paid miner earns; the unpaid one still does not.
        later = Validator(VALIDATOR, chain, ValidatorConfig(**{**config.__dict__, "policy": NO_WAIT}))
        weights = await later.run_round()
        assert weights[1] == pytest.approx(2 * unit) and 2 not in weights and 5 not in weights
        assert weights[4] == pytest.approx(1 - 2 * unit)
        assert chain.scanned[-1] == (101, 100) or len(chain.scanned) == 1  # nothing new to read

        # 3 TAO is under one RTX 4090 fee; with 4.4 more paid through the
        # platform, 7.4 covers both of its GPUs and it is listed and earns.
        platform_file(tmp_path, activation_credit={unpaid.ss58_address: 4.4})
        weights = await later.run_round()
        assert weights[1] == pytest.approx(2 * unit) and weights[2] == pytest.approx(2 * unit)
        status = json.loads((tmp_path / "status.json").read_text())
        assert status["miners"][unpaid.ss58_address]["activation_credit_tao"] == pytest.approx(0)
        assert status["activation_fees_tao"]["h100-80g"] == 10.0
    finally:
        for server in (np_, nu, pricey):
            await server.close()


async def test_unexpected_node_error_is_signed_and_order_can_be_asked_about(tmp_path, monkeypatch):
    """A crash while taking an order still gives a signed answer, and the
    platform can then ask whether the order was taken."""
    hotkey = key("Crashy")
    server = await start_node(tmp_path, "crashy", hotkey)
    monkeypatch.setattr(Node, "_save_orders", lambda self: (_ for _ in ()).throw(OSError("disk full")))
    try:
        async with aiohttp.ClientSession() as session:
            platform = NodeClient(session, PLATFORM, f"http://127.0.0.1:{server.port}", hotkey.ss58_address)
            reply = await platform.call("POST", "/v1/orders", {"order_id": "wr-9", "gpus": 1, "ssh_keys": ["ssh-ed25519 AAAA test"]})
            assert reply.status == 500 and "disk full" in reply.data["error"]
            monkeypatch.undo()
            # An order the node could not save was not taken: no GPUs held.
            assert (await platform.call("GET", "/v1/orders/wr-9")).status == 404
            inventory = (await platform.call("GET", "/v1/inventory")).data
            assert not any(g["rented"] for g in inventory["gpus"])
    finally:
        await server.close()
