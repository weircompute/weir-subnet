"""Whole subnet lifecycle on a local chain, using only throwaway dev keys.

    docker run -d --name weir-localnet -p 127.0.0.1:19944:9944 \\
        ghcr.io/raofoundation/subtensor-localnet:testnet
    python scripts/localnet_e2e.py ws://127.0.0.1:19944

Steps: fund owner and miner from the dev account, register a subnet, activate
it, register the miner, start a node with a simulated GPU, publish its
address, run one validator round with the owner hotkey, wait for the epoch and
read back weights and incentive.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

import bittensor as bt
from aiohttp import web
from bittensor.keyfiles import Keypair

from weir_subnet.node import gpus
from weir_subnet.node.server import Node, NodeConfig
from weir_subnet.validator.chain import BittensorChain
from weir_subnet.validator.scoring import Policy
from weir_subnet.validator.validator import Validator, ValidatorConfig

NODE_PORT = 18091
# The chain refuses loopback addresses, so the node binds to the Docker bridge.
NODE_IP = "172.17.0.1"


def step(name):
    print(f"\n== {name}", flush=True)


def ok(result, what):
    if not result.success:
        raise SystemExit(f"{what} failed: {result.error!r}")
    print(f"   {what}: ok", flush=True)
    return result


def wallet(root: Path, name: str) -> bt.Wallet:
    w = bt.Wallet(name=name, hotkey="default", path=str(root))
    w.create_new_coldkey(use_password=False, suppress=True, overwrite=True)
    w.create_new_hotkey(use_password=False, suppress=True, overwrite=True)
    return w


async def main(endpoint: str) -> None:
    root = Path(tempfile.mkdtemp(prefix="weir-localnet-"))
    sub = bt.Subtensor(endpoint)
    alice = Keypair.create_from_uri("//Alice")
    owner, miner = wallet(root, "owner"), wallet(root, "miner")

    step("fund")
    ok(sub.execute(bt.Transfer(dest_ss58=owner.coldkeypub.ss58_address, amount_tao=50000), alice), "owner funded")  # lock cost doubles per subnet
    ok(sub.execute(bt.Transfer(dest_ss58=miner.coldkeypub.ss58_address, amount_tao=100), alice), "miner funded")

    step("register subnet")
    result = ok(sub.execute(bt.RegisterSubnet(), owner), "register-subnet")
    netuid = getattr(result, "netuid", None)
    if netuid is None:
        netuid = max(s.netuid for s in sub.subnets.all())
    print(f"   netuid {netuid}")

    step("activate (start-call)")
    for attempt in range(120):
        result = sub.execute(bt.StartCall(netuid=netuid), owner)
        if result.success:
            print("   start-call: ok")
            break
        if attempt % 10 == 0:
            print(f"   waiting: {result.error.code}", flush=True)
        time.sleep(2)
    else:
        raise SystemExit("start-call never succeeded")

    step("register miner")
    for attempt in range(5):  # right after activation the first try can be refused
        result = sub.execute(bt.BurnedRegister(netuid=netuid), miner)
        if result.success:
            break
        print(f"   retrying: {result.error!r}", flush=True)
        time.sleep(3)
    ok(result, "burned-register")

    step("start node and publish its address")
    config = NodeConfig(
        hotkey=miner.hotkey, validators={owner.hotkey.ss58_address}, platform=set(),
        gpus=gpus.simulated("RTX 4090:24564:1", miner.hotkey.ss58_address), simulated=True,
        price_usd_per_gpu_hour="0.35", state_dir=root / "node",
    )
    runner = web.AppRunner(Node(config).app())
    await runner.setup()
    await web.TCPSite(runner, NODE_IP, NODE_PORT).start()
    ok(sub.execute(bt.ServeAxon(netuid=netuid, ip=NODE_IP, port=NODE_PORT, version=100), miner), "serve-axon")

    step("validator round (owner hotkey)")
    chain = BittensorChain(endpoint, netuid, "owner", "default", str(root))
    snapshot = chain.snapshot()
    print(f"   metagraph: {[(n.uid, n.hotkey[:8], n.axon) for n in snapshot.neurons]}, owner uid {snapshot.owner_uid}")
    validator = Validator(owner.hotkey, chain, ValidatorConfig(
        state_file=root / "validator.json", summary_file=root / "summary.json",
        challenge_n=256, max_pairs=4, timing=False, weights_every=0, target_capacity=32, policy=Policy(require_sysbox=False, min_disk_ratio=0),
    ))
    weights = None
    for attempt in range(20):  # the weights rate limit is already running after registration
        weights = await validator.run_round()
        if weights is not None:
            break
        print("   weights refused for now, next round in 10s", flush=True)
        await asyncio.sleep(10)
    record = validator.miners[miner.hotkey.ss58_address]
    print(f"   miner GPU: {[g.last_result for g in record.gpus.values()]}")
    print(f"   weights sent: {weights}")

    step("wait for the epoch, then read back")
    miner_uid = next(n.uid for n in snapshot.neurons if n.hotkey == miner.hotkey.ss58_address)
    deadline = time.time() + 600
    while time.time() < deadline:
        raw = sub.read("metagraph", netuid=netuid)
        incentive = raw["incentives"][miner_uid]
        rows = sub.weights.weights(netuid=netuid)
        if incentive and float(incentive) > 0:
            print(f"   on-chain weights: {rows}")
            print(f"   miner incentive {incentive}, owner incentive {raw['incentives'][snapshot.owner_uid]}")
            break
        time.sleep(5)
    else:
        print(f"   no incentive after 10 minutes; weights on chain: {sub.weights.weights(netuid=netuid)}")
    await runner.cleanup()
    print(f"\nkeys and state in {root}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:19944"))
