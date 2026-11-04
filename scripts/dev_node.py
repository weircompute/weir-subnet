"""A development node that weir-api can rent from, with no chain.

Starts a node (simulated GPUs, real Docker containers), runs one validator
round against it so the validator writes sellable.json, then keeps serving.

    python scripts/dev_node.py --platform <weir-api platform address> \\
        --sellable /tmp/sellable.json --port 18092 --image weir-test-sshd

Used by weir-api's tests/subnet_nodes.rs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import tempfile
from pathlib import Path

from aiohttp import web
from bittensor_core import Keypair

from weir_subnet.node import gpus
from weir_subnet.node.runtime import DockerRuntime
from weir_subnet.node.server import Node, NodeConfig
from weir_subnet.validator.chain import Neuron, Snapshot
from weir_subnet.validator.scoring import Policy
from weir_subnet.validator.validator import Validator, ValidatorConfig


class OneNodeChain:
    def __init__(self, owner: str, miner: str, axon: str):
        self.neurons = [Neuron(0, owner, None, True), Neuron(1, miner, axon, False)]

    def snapshot(self):
        return Snapshot(1, self.neurons[0].hotkey, self.neurons)

    def set_weights(self, weights):
        print("weights", weights, flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--sellable", required=True)
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--image", default="weir-test-sshd")
    parser.add_argument("--price", default="0.30")
    parser.add_argument("--gpus", default="RTX 4090:24564:2")
    parser.add_argument("--payout-address", default="0x00000000000000000000000000000000000dEaD1")
    args = parser.parse_args()

    miner = Keypair.create_from_uri("//WeirDevMiner")
    validator_key = Keypair.create_from_uri("//WeirDevValidator")
    state = Path(tempfile.mkdtemp(prefix="weir-dev-node-"))
    runtime = DockerRuntime({"ubuntu-2204-cuda124": args.image, "ubuntu-2204": args.image}, bind_ip="127.0.0.1",
                            use_gpus=False, cpus_per_gpu=8, memory_gib_per_gpu=1)
    config = NodeConfig(
        hotkey=miner, validators={validator_key.ss58_address}, platform={args.platform},
        gpus=gpus.simulated(args.gpus, miner.ss58_address), simulated=True, price_usd_per_gpu_hour=args.price,
        state_dir=state / "node", public_host="127.0.0.1", runtime=runtime, ssh_ports=(42960, 42990),
        images=["ubuntu-2204", "ubuntu-2204-cuda124"], cpus_per_gpu=8, memory_gib_per_gpu=32,
        payout_address=args.payout_address,
    )
    runner = web.AppRunner(Node(config).app())
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()

    chain = OneNodeChain(Keypair.create_from_uri("//WeirDevOwner").ss58_address, miner.ss58_address, f"127.0.0.1:{args.port}")
    platform = state / "platform.json"  # a cap above the dev price, so the node is sellable
    platform.write_text(json.dumps({"generated_at": time.time(), "caps": {"rtx4090-24g": 1.0, "a100-80g": 1.0, "h100-80g": 1.4}}))
    validator = Validator(validator_key, chain, ValidatorConfig(
        state_file=state / "validator.json", sellable_file=Path(args.sellable),
        challenge_n=128, max_pairs=2, timing=False, weights_every=0, platform_file=platform,
        policy=Policy(require_sysbox=False, min_disk_ratio=0, probation_seconds=0),
    ))
    await validator.run_round()
    listed = json.loads(Path(args.sellable).read_text())["nodes"]
    print(json.dumps({"ready": True, "miner": miner.ss58_address, "nodes": len(listed)}), flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
