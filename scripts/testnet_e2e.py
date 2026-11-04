"""Whole subnet lifecycle on the public testnet, one step per run so it can
resume. Every wallet it touches, the owner included, is for testnet only:
never reuse one on mainnet. The miner node runs with a simulated GPU on the
Docker bridge address, so no public IP is published on chain.

    python scripts/testnet_e2e.py <state dir> <step>

Steps: subnet, start, miner, round, check; then the economics:
  extra   reserve wallet registered, fee wallet created
  fee     the miner pays its activation fee to the fee wallet
  round2  a validator round with reserve, fee and price caps
Needs WEIR_OWNER_PASSWORD_FILE: a file with a "coldkey password: ..." line.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import bittensor as bt
from aiohttp import web

from weir_subnet.node import gpus
from weir_subnet.node.server import Node, NodeConfig
from weir_subnet.validator.chain import BittensorChain
from weir_subnet.validator.scoring import Policy
from weir_subnet.validator.validator import Validator, ValidatorConfig

ENDPOINT = "wss://test.finney.opentensor.ai:443"
OWNER_PATH = "/root/.bittensor/wallets"
NODE_IP = "172.17.0.1"
NODE_PORT = 18091
MINER_FUNDS_TAO = 0.2
EXTRA_FUNDS_TAO = 0.02  # registration and transaction fees for A and B
TEST_FEE_TAO = 0.01  # stands in for the per-GPU activation fee on testnet (set through the platform file)


def ok(result, what):
    if not result.success:
        raise SystemExit(f"{what} failed: {result.error!r}")
    print(f"   {what}: ok", flush=True)
    return result


def owner_wallet() -> bt.Wallet:
    w = bt.Wallet(name="weir-owner", hotkey="default", path=OWNER_PATH)
    text = Path(os.environ["WEIR_OWNER_PASSWORD_FILE"]).read_text()
    password = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.lower().startswith("coldkey password:"))
    w.coldkey_file.save_password_to_env(password)  # only in this process
    return w


def miner_wallet(state_dir: Path) -> bt.Wallet:
    w = bt.Wallet(name="weir-test-miner", hotkey="default", path=str(state_dir / "wallets"))
    if not w.coldkeypub_file.exists_on_device():
        w.create_new_coldkey(use_password=False, suppress=True, overwrite=False)
        w.create_new_hotkey(use_password=False, suppress=True, overwrite=False)
    return w


def platform_file(state_dir: Path) -> Path:
    """Price caps for the simulated card, so the test miner can earn idle pay."""
    path = state_dir / "platform.json"
    path.write_text(json.dumps({"generated_at": time.time(), "caps": {"rtx4090-24g": 1.0}}))
    return path


def throwaway(state_dir: Path, name: str) -> bt.Wallet:
    w = bt.Wallet(name=name, hotkey="default", path=str(state_dir / "wallets"))
    if not w.coldkeypub_file.exists_on_device():
        w.create_new_coldkey(use_password=False, suppress=True, overwrite=False)
        w.create_new_hotkey(use_password=False, suppress=True, overwrite=False)
    return w


def load(state_dir: Path) -> dict:
    path = state_dir / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save(state_dir: Path, state: dict) -> None:
    (state_dir / "state.json").write_text(json.dumps(state, indent=2))


async def main(state_dir: Path, step: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    state = load(state_dir)
    sub = bt.Subtensor(ENDPOINT)
    owner = owner_wallet()

    if step == "subnet":
        if "netuid" in state:
            print(f"   already registered: netuid {state['netuid']}")
            return
        before = {s.netuid for s in sub.subnets.all()}
        result = ok(sub.execute(bt.RegisterSubnet(), owner), "register-subnet")  # no identity
        netuid = getattr(result, "netuid", None)
        if netuid is None:
            owned = [s.netuid for s in sub.subnets.all() if s.netuid not in before]
            netuid = max(owned)
        state["netuid"] = netuid
        save(state_dir, state)
        print(f"   netuid {netuid}")

    elif step == "start":
        result = sub.execute(bt.StartCall(netuid=state["netuid"]), owner)
        print(f"   start-call: {'ok' if result.success else result.error!r}")
        if result.success:
            state["started"] = True
            save(state_dir, state)

    elif step == "miner":
        miner = miner_wallet(state_dir)
        if sub.read("balance", coldkey_ss58=miner.coldkeypub.ss58_address).tao < MINER_FUNDS_TAO / 2:
            ok(sub.execute(bt.Transfer(dest_ss58=miner.coldkeypub.ss58_address, amount_tao=MINER_FUNDS_TAO), owner), "miner funded")
        for attempt in range(5):
            result = sub.execute(bt.BurnedRegister(netuid=state["netuid"]), miner)
            if result.success:
                break
            print(f"   retrying: {result.error!r}", flush=True)
            time.sleep(6)
        ok(result, "burned-register")
        state["miner_hotkey"] = miner.hotkey.ss58_address
        save(state_dir, state)

    elif step == "round":
        miner = miner_wallet(state_dir)
        netuid = state["netuid"]
        config = NodeConfig(
            hotkey=miner.hotkey, validators={owner.hotkey.ss58_address}, platform=set(),
            gpus=gpus.simulated("RTX 4090:24564:1", miner.hotkey.ss58_address), simulated=True,
            price_usd_per_gpu_hour="0.35", state_dir=state_dir / "node",
        )
        runner = web.AppRunner(Node(config).app())
        await runner.setup()
        await web.TCPSite(runner, NODE_IP, NODE_PORT).start()
        ok(sub.execute(bt.ServeAxon(netuid=netuid, ip=NODE_IP, port=NODE_PORT, version=100), miner), "serve-axon")
        chain = BittensorChain(ENDPOINT, netuid, "weir-owner", "default", OWNER_PATH)
        snapshot = chain.snapshot()
        print(f"   metagraph: {[(n.uid, n.hotkey[:8], n.axon) for n in snapshot.neurons]}, owner uid {snapshot.owner_uid}")
        validator = Validator(owner.hotkey, chain, ValidatorConfig(
            state_file=state_dir / "validator.json", summary_file=state_dir / "summary.json",
            challenge_n=256, max_pairs=4, timing=False, weights_every=0, target_capacity=32,
            policy=Policy(require_sysbox=False, min_disk_ratio=0, probation_seconds=0),
            platform_file=platform_file(state_dir),
        ))
        weights = None
        for attempt in range(30):  # the weights rate limit may still be running
            weights = await validator.run_round()
            if weights is not None:
                break
            print("   weights refused for now, next round in 20s", flush=True)
            await asyncio.sleep(20)
        record = validator.miners[miner.hotkey.ss58_address]
        print(f"   miner GPU: {[g.last_result for g in record.gpus.values()]}")
        print(f"   weights sent: {weights}")
        await runner.cleanup()

    elif step == "extra":
        netuid = state["netuid"]
        reserve, fee = throwaway(state_dir, "weir-test-reserve"), throwaway(state_dir, "weir-test-fee")
        registered = set(sub.read("metagraph", netuid=netuid)["hotkeys"])
        for label, w in (("reserve", reserve),):
            if w.hotkey.ss58_address in registered:
                print(f"   {label} already registered")
                continue
            if sub.read("balance", coldkey_ss58=w.coldkeypub.ss58_address).tao < EXTRA_FUNDS_TAO / 2:
                ok(sub.execute(bt.Transfer(dest_ss58=w.coldkeypub.ss58_address, amount_tao=EXTRA_FUNDS_TAO), owner), f"{label} funded")
            for attempt in range(5):
                result = sub.execute(bt.BurnedRegister(netuid=netuid), w)
                if result.success:
                    break
                print(f"   retrying: {result.error!r}", flush=True)
                time.sleep(12)
            ok(result, f"{label} registered")
        state.update(reserve_hotkey=reserve.hotkey.ss58_address, fee_coldkey=fee.coldkeypub.ss58_address)
        save(state_dir, state)

    elif step == "fee":
        miner = miner_wallet(state_dir)
        state["fee_block"] = sub.block
        save(state_dir, state)
        ok(sub.execute(bt.Transfer(dest_ss58=state["fee_coldkey"], amount_tao=TEST_FEE_TAO), miner), f"activation fee {TEST_FEE_TAO} sent")

    elif step == "round2":
        miner = miner_wallet(state_dir)
        netuid = state["netuid"]
        config = NodeConfig(
            hotkey=miner.hotkey, validators={owner.hotkey.ss58_address}, platform=set(),
            gpus=gpus.simulated("RTX 4090:24564:1", miner.hotkey.ss58_address), simulated=True,
            price_usd_per_gpu_hour="0.35", state_dir=state_dir / "node",
        )
        runner = web.AppRunner(Node(config).app())
        await runner.setup()
        await web.TCPSite(runner, NODE_IP, NODE_PORT).start()
        platform = state_dir / "platform.json"
        platform.write_text(json.dumps({"generated_at": time.time(), "caps": {"rtx4090-24g": 1.0, "h100-80g": 1.4},
                                        "fees": {"rtx4090-24g": TEST_FEE_TAO, "h100-80g": TEST_FEE_TAO}}))
        rentals = state_dir / "rentals.json"
        rentals.write_text(json.dumps({miner.hotkey.ss58_address: {"gpu_hours": 10, "delivery": 1.0}}))
        chain = BittensorChain(ENDPOINT, netuid, "weir-owner", "default", OWNER_PATH)
        validator = Validator(owner.hotkey, chain, ValidatorConfig(
            state_file=state_dir / "validator2.json", summary_file=state_dir / "summary2.json", status_file=state_dir / "status2.json",
            challenge_n=256, max_pairs=4, timing=False, weights_every=0, target_capacity=32,
            policy=Policy(require_sysbox=False, min_disk_ratio=0, probation_seconds=0),
            platform_file=platform, rentals_file=rentals,
            reserve_hotkey=state["reserve_hotkey"],
            fee_address=state["fee_coldkey"], fee_start_block=state["fee_block"],
        ))
        weights = None
        for attempt in range(30):  # catches up on fee blocks, then the weights rate limit
            weights = await validator.run_round()
            if weights is not None and validator.fee_scanned_block >= sub.block - 30:
                break
            print(f"   fees read to block {validator.fee_scanned_block}, weights {weights}; next round in 20s", flush=True)
            await asyncio.sleep(20)
        print(f"   fees paid: {validator.fees_paid}")
        print(f"   miner fee paid: {validator.miners[miner.hotkey.ss58_address].fee_paid}")
        print(f"   weights sent: {weights}")
        await runner.cleanup()

    elif step == "check":
        netuid = state["netuid"]
        raw = sub.read("metagraph", netuid=netuid)
        hotkeys = raw["hotkeys"]
        print(f"   block {sub.block}, neurons {len(hotkeys)}")
        for uid, hotkey in enumerate(hotkeys):
            print(f"   uid {uid} {hotkey[:8]} incentive {raw['incentives'][uid]} dividends {raw['dividends'][uid]} emission {raw['emission'][uid]}")
        print(f"   weights on chain: {sub.weights.weights(netuid=netuid)}")
        print(f"   owner balance {sub.read("balance", coldkey_ss58=owner.coldkeypub.ss58_address)}")

    else:
        raise SystemExit("step must be subnet, start, miner, round, check, extra, fee or round2")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
