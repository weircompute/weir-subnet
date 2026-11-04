"""Command line: `weir-node` for miners, `weir-validator` for validators."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from . import network

DEFAULT_WALLETS = os.path.expanduser("~/.bittensor/wallets")


def _hotkey(args) -> "Keypair":  # noqa: F821
    import bittensor as bt

    # Only the hotkey is loaded. Nothing here ever asks for the coldkey.
    return bt.Wallet(name=args.wallet, hotkey=args.hotkey, path=args.wallet_path).hotkey


def _wallet_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wallet", required=True, help="wallet name")
    p.add_argument("--hotkey", default="default")
    p.add_argument("--wallet-path", default=DEFAULT_WALLETS)


def node_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="weir-node")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="serve the node API")
    _wallet_args(run)
    run.add_argument("--port", type=int, default=8091)
    run.add_argument("--host", default="0.0.0.0")
    run.add_argument("--validators", default=network.VALIDATORS, help="comma-separated validator hotkeys allowed to challenge")
    run.add_argument("--platform", default=network.PLATFORM, help="comma-separated hotkeys allowed to manage orders")
    run.add_argument("--price", default="0", help="ask, USD per GPU-hour")
    run.add_argument("--public-host", default="", help="address tenants will SSH to")
    run.add_argument("--state-dir", default="/var/lib/weir-node")
    run.add_argument("--simulate", default="", help='development only, e.g. "RTX 4090:24564:2"')
    run.add_argument("--runtime", choices=["docker", "none"], default="docker", help="none: track orders only (development)")
    run.add_argument("--images", default=network.IMAGES, help="Weir image id = Docker image, comma-separated")
    run.add_argument("--cpus-per-gpu", type=float, default=8.0)
    run.add_argument("--memory-gib-per-gpu", type=int, default=48)
    run.add_argument("--docker-runtime", default="sysbox-runc", help='"sysbox-runc" (needed for idle pay) or "" for plain Docker')
    run.add_argument("--bind-ip", default="0.0.0.0", help="address tenant SSH ports listen on")
    run.add_argument("--ssh-ports", default="42000-42999")
    run.add_argument("--payout-address", default="", help="stablecoin address that receives your rent share")
    run.add_argument("--name", default="main", help="this machine's name, if you run several")
    run.add_argument("--metrics-port", type=int, default=0, help="serve Prometheus metrics on 127.0.0.1:<port> (off when 0)")
    run.add_argument("--fleet", default="", help="on the machine at your chain address: name=url of every machine, comma-separated (include this one)")

    announce = sub.add_parser("announce", help="publish this node's address on chain (hotkey-signed)")
    _wallet_args(announce)
    announce.add_argument("--network", default=network.NETWORK)
    announce.add_argument("--netuid", type=int, default=network.NETUID)
    announce.add_argument("--ip", required=True)
    announce.add_argument("--port", type=int, default=8091)

    selftest = sub.add_parser("selftest", help="run the validator's checks on this machine")
    selftest.add_argument("--simulate", default="", help='development only, e.g. "RTX 4090:24564:2"')
    selftest.add_argument("--n", type=int, default=8192, help="matrix size (the validator uses 8192)")
    selftest.add_argument("--image", default="nvidia/cuda:12.4.1-runtime-ubuntu22.04", help="a tenant image to check GPUs inside a container")
    selftest.add_argument("--docker-runtime", default="", help='runtime to check, default "sysbox-runc"')
    selftest.add_argument("--payout-address", default="")
    selftest.add_argument("--json", action="store_true", help="print measured numbers (used to calibrate time limits)")

    for name, text in (("summary", "balance, payable amount, payout address"), ("earnings", "statement of every earning, penalty and payout"),
                       ("payouts", "payouts and their transaction hashes"), ("status", "why each machine and GPU is or is not sold")):
        cmd = sub.add_parser(name, help=text)
        _wallet_args(cmd)
        cmd.add_argument("--platform-url", default=network.PLATFORM_URL, help="Weir platform address")
        cmd.add_argument("--platform", default=network.PLATFORM, help="the platform's hotkey address")
        if name == "earnings":
            cmd.add_argument("--csv", action="store_true", help="CSV instead of JSON")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "selftest":
        from .node import gpus as gpu_mod
        from .node import selftest as st

        found = gpu_mod.simulated(args.simulate, "selftest") if args.simulate else gpu_mod.detect()
        checks = asyncio.run(st.run(found, n=args.n, simulated=bool(args.simulate), image=args.image,
                                    docker_runtime=args.docker_runtime, payout_address=args.payout_address))
        print(st.report(checks, args.json))
        raise SystemExit(1 if any(c.result == "FAIL" for c in checks) else 0)

    if args.command in ("summary", "earnings", "payouts", "status"):
        print(asyncio.run(_miner_report(args)))
        return

    if args.command == "announce":
        import bittensor as bt

        from .protocol import AGENT_VERSION

        wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey, path=args.wallet_path)
        version = int("".join(f"{int(p):02d}" for p in AGENT_VERSION.split(".")))
        result = bt.Subtensor(args.network).execute(bt.ServeAxon(netuid=args.netuid, ip=args.ip, port=args.port, version=version), wallet)
        print("published" if result.success else f"failed: {result.error.code} {result.error.remediation}")
        return

    from aiohttp import web

    from .node import gpus
    from .node.runtime import DockerRuntime, StateOnlyRuntime
    from .node.server import Node, NodeConfig

    keypair = _hotkey(args)
    found = gpus.simulated(args.simulate, keypair.ss58_address) if args.simulate else gpus.detect()
    if not found:
        raise SystemExit("no GPUs found (is the NVIDIA driver installed?)")
    if not args.name.isidentifier():
        raise SystemExit("--name may use letters, digits and _ only")
    if args.payout_address and not (args.payout_address.isalnum() and 20 <= len(args.payout_address) <= 100):
        raise SystemExit("--payout-address does not look like a wallet address")
    images = dict(pair.split("=", 1) for pair in args.images.split(",") if "=" in pair)
    if args.runtime == "docker":
        if not images:
            raise SystemExit("--images is required with the docker runtime")
        runtime = DockerRuntime(images, bind_ip=args.bind_ip, docker_runtime=args.docker_runtime, use_gpus=not args.simulate,
                                cpus_per_gpu=args.cpus_per_gpu, memory_gib_per_gpu=args.memory_gib_per_gpu)
    else:
        runtime = StateOnlyRuntime()
    low, high = (int(p) for p in args.ssh_ports.split("-"))
    config = NodeConfig(
        hotkey=keypair,
        validators={v for v in args.validators.split(",") if v},
        platform={v for v in args.platform.split(",") if v},
        gpus=found, simulated=bool(args.simulate), price_usd_per_gpu_hour=args.price,
        state_dir=Path(args.state_dir), public_host=args.public_host,
        runtime=runtime, ssh_ports=(low, high), images=sorted(images),
        cpus_per_gpu=args.cpus_per_gpu, memory_gib_per_gpu=args.memory_gib_per_gpu,
        payout_address=args.payout_address, name=args.name,
        fleet=[tuple(pair.split("=", 1)) for pair in args.fleet.split(",") if "=" in pair],
    )
    logging.info("node %s with %d GPU(s) on port %d", keypair.ss58_address, len(found), args.port)
    node = Node(config)
    if not args.metrics_port:
        web.run_app(node.app(), host=args.host, port=args.port, print=None)
        return

    async def serve() -> None:
        for app, host, port in ((node.app(), args.host, args.port), (node.metrics_app(), "127.0.0.1", args.metrics_port)):
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, host, port).start()
        await asyncio.Event().wait()

    asyncio.run(serve())


async def _miner_report(args) -> str:
    """Asks the platform about this miner, signed with the hotkey."""
    import json

    import aiohttp
    import bittensor as bt

    from .validator.codes import EXPLAIN

    path = f"/v1/miner/{args.command}" + ("?format=csv" if getattr(args, "csv", False) else "")
    headers = bt.http_auth.sign(_hotkey(args), method="GET", path=path, receiver_ss58=args.platform)
    async with aiohttp.ClientSession() as session:
        async with session.get(args.platform_url.rstrip("/") + path, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            text = await response.text()
            if response.status != 200:
                raise SystemExit(f"platform refused ({response.status}): {text}")
    if args.command != "status":
        return text if getattr(args, "csv", False) else json.dumps(json.loads(text), indent=1)
    miner = json.loads(text).get("miner")
    if not miner:
        return "The validator has no record of this hotkey yet."
    lines = [f"Activation credit: {miner.get('activation_credit_tao', 0)} TAO"]
    fees = miner.get("activation_fees_tao") or {}
    if fees:
        lines.append("Activation fee per GPU: " + ", ".join(f"{k} {v} TAO" for k, v in fees.items()))
    for node in miner.get("nodes", []):
        lines.append(f"{node['name']}: {node['code']} - {EXPLAIN.get(node['code'], '')} {node.get('detail') or ''}".rstrip())
        for gpu in node.get("gpus", []):
            detail = f" ({gpu['detail']})" if gpu.get("detail") else ""
            lines.append(f"  GPU {gpu['index']} {gpu.get('model') or '?'}: {gpu['code']} - {EXPLAIN.get(gpu['code'], '')}{detail}")
    return "\n".join(lines)


def validator_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="weir-validator")
    _wallet_args(parser)
    parser.add_argument("--network", default=network.NETWORK)
    parser.add_argument("--netuid", type=int, default=network.NETUID)
    parser.add_argument("--state-file", default="/var/lib/weir-validator/state.json")
    parser.add_argument("--summary-file", default="/var/lib/weir-validator/summary.json")
    parser.add_argument("--rentals-file", default="", help="settled rental stats from the platform")
    parser.add_argument("--sellable-file", default="/var/lib/weir-validator/sellable.json", help="verified nodes for the platform")
    parser.add_argument("--status-file", default="/var/lib/weir-validator/status.json", help="reason codes per node and GPU, for miners")
    parser.add_argument("--round-seconds", type=float, default=300)
    parser.add_argument("--weights-every", type=float, default=1200)
    parser.add_argument("--target-capacity", type=float, default=1134.0, help="idle weighted GPUs that fill the idle pool")
    parser.add_argument("--idle-cap-gpus", type=int, default=64, help="idle GPUs per model paid in full")
    parser.add_argument("--probation-days", type=float, default=14.0, help="days of passes without a gap before idle pay starts")
    parser.add_argument("--platform-file", default="", help="price caps, per-model weight and fee overrides, and activation credit paid through the platform")
    parser.add_argument("--reserve-hotkey", default="", help="receives what the pools leave (default: the owner UID)")
    parser.add_argument("--fee-address", default=network.FEE_ADDRESS, help="coldkey that receives activation fees (empty: no fee)")
    parser.add_argument("--fee-start-block", type=int, default=0, help="first block to read fees from (older blocks need an archive endpoint)")
    parser.add_argument("--dev-no-idle-gates", action="store_true", help="development only: skip the Sysbox and disk gates")
    parser.add_argument("--challenge-n", type=int, default=0, help="fixed matrix size; default varies 6144-8192")
    parser.add_argument("--dev-timing-off", action="store_true", help="development only: accept simulated GPUs")
    parser.add_argument("--once", action="store_true", help="run one round and set weights, then exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .validator.chain import BittensorChain
    from .validator.validator import Validator, ValidatorConfig
    from .validator.scoring import Policy

    chain = BittensorChain(args.network, args.netuid, args.wallet, args.hotkey, args.wallet_path)
    config = ValidatorConfig(
        state_file=Path(args.state_file), summary_file=Path(args.summary_file),
        rentals_file=Path(args.rentals_file) if args.rentals_file else None,
        sellable_file=Path(args.sellable_file) if args.sellable_file else None,
        status_file=Path(args.status_file) if args.status_file else None,
        weights_every=0 if args.once else args.weights_every, target_capacity=args.target_capacity,
        platform_file=Path(args.platform_file) if args.platform_file else None,
        reserve_hotkey=args.reserve_hotkey,
        fee_address=args.fee_address, fee_start_block=args.fee_start_block,
        challenge_n=args.challenge_n, timing=not args.dev_timing_off,
        policy=Policy(idle_cap_gpus=args.idle_cap_gpus, probation_seconds=args.probation_days * 86400,
                      require_sysbox=not args.dev_no_idle_gates, min_disk_ratio=0 if args.dev_no_idle_gates else 1.5),
    )
    validator = Validator(_hotkey(args), chain, config)
    if args.once:
        print(asyncio.run(validator.run_round()))
    else:
        asyncio.run(validator.run_forever(args.round_seconds))
