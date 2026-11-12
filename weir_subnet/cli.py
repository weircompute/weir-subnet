"""Miner command line."""

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
    run.add_argument("--host", default="127.0.0.1", help="control API bind address")
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
    run.add_argument("--fleet", default="", help="name=url of each machine, comma-separated")

    selftest = sub.add_parser("selftest", help="run local hardware diagnostics")
    selftest.add_argument("--simulate", default="", help='development only, e.g. "RTX 4090:24564:2"')
    selftest.add_argument("--n", type=int, default=8192, help="local diagnostic matrix size")
    selftest.add_argument("--image", default="nvidia/cuda:12.4.1-runtime-ubuntu22.04", help="a tenant image to check GPUs inside a container")
    selftest.add_argument("--docker-runtime", default="", help='runtime to check, default "sysbox-runc"')
    selftest.add_argument("--payout-address", default="")
    selftest.add_argument("--json", action="store_true", help="print measured numbers (used to calibrate time limits)")

    for name, text in (("summary", "balance, payable amount, payout address"), ("earnings", "statement of every earning, penalty and payout"),
                       ("payouts", "payouts and their transaction hashes"),
                       ("quote", "current activation fees and price limits, held for this hotkey for 7 days")):
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

    if args.command in ("summary", "earnings", "payouts", "quote"):
        print(asyncio.run(_miner_report(args)))
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

    path = f"/v1/miner/{args.command}" + ("?format=csv" if getattr(args, "csv", False) else "")
    headers = bt.http_auth.sign(_hotkey(args), method="GET", path=path, receiver_ss58=args.platform)
    async with aiohttp.ClientSession() as session:
        async with session.get(args.platform_url.rstrip("/") + path, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            text = await response.text()
            if response.status != 200:
                raise SystemExit(f"platform refused ({response.status}): {text}")
    if args.command == "quote":
        return _quote_text(json.loads(text))
    return text if getattr(args, "csv", False) else json.dumps(json.loads(text), indent=1)


def _quote_text(quote: dict) -> str:
    lines = [f"{'Card':<14} {'Activation fee':>15} {'Price per GPU-hour':>22}"]
    for card in quote.get("cards", []):
        fee = card.get("activationFeeTao")
        fee = f"{fee} TAO" if fee is not None else "not open"
        price = f"${card['priceFloorUsd']} - ${card['priceCapUsd']}" if card.get("priceCapUsd") else "-"
        lines.append(f"{card['name']:<14} {fee:>15} {price:>22}")
    lines.append("")
    if quote.get("hotkey"):
        lines.append(f"Quote {quote['quoteId']} for {quote['hotkey']}, held until {quote['expiresAt']}:")
        lines.append("GPUs this hotkey activates before then pay at most these fees.")
    lines.append(f"Pay from this hotkey's coldkey to {quote.get('feeAddress')}.")
    return "\n".join(lines)
