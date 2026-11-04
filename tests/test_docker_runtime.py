"""DockerRuntime against a real Docker engine. Opt in with WEIR_DOCKER_TESTS=1.

Needs the image `weir-test-sshd` (Debian + openssh-server) and an ssh client.
Containers are named weir-order-*, bound to 127.0.0.1, and removed at the end.
"""

import asyncio
import os
import subprocess
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestServer
from bittensor_core import Keypair

from weir_subnet.node import gpus
from weir_subnet.node.runtime import DockerRuntime
from weir_subnet.node.server import Node, NodeConfig
from weir_subnet.validator.client import NodeClient

pytestmark = pytest.mark.skipif(os.environ.get("WEIR_DOCKER_TESTS") != "1", reason="needs Docker (WEIR_DOCKER_TESTS=1)")

IMAGE = "weir-test-sshd"
PLATFORM, NODE = Keypair.create_from_uri("//WeirPlatform"), Keypair.create_from_uri("//DockerNode")


def ssh(key: Path, port: int, command: str, timeout=20) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", str(key), "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "root@127.0.0.1", command],
        capture_output=True, text=True, timeout=timeout,
    )


def docker(*args) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True).stdout.strip()


async def wait_running(client, order_id, seconds=90):
    for _ in range(seconds):
        view = (await client.call("GET", f"/v1/orders/{order_id}")).data
        if view["state"] != "starting":
            return view
        await asyncio.sleep(1)
    return view


async def test_order_gets_an_isolated_ssh_container(tmp_path):
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "id")], check=True)
    pub = (tmp_path / "id.pub").read_text().strip()
    runtime = DockerRuntime({IMAGE: IMAGE}, bind_ip="127.0.0.1", use_gpus=False, cpus_per_gpu=1, memory_gib_per_gpu=1)
    config = NodeConfig(hotkey=NODE, validators=set(), platform={PLATFORM.ss58_address},
                        gpus=gpus.simulated("RTX 4090:24564:2", "docker"), simulated=True,
                        state_dir=tmp_path / "node", public_host="127.0.0.1", runtime=runtime, ssh_ports=(42900, 42950))
    server = TestServer(Node(config).app(), host="127.0.0.1")
    await server.start_server()
    orders = ["dock-a", "dock-b"]
    try:
        async with aiohttp.ClientSession() as session:
            platform = NodeClient(session, PLATFORM, f"http://127.0.0.1:{server.port}", NODE.ss58_address)
            for order_id in orders:
                placed = await platform.call("POST", "/v1/orders", {"order_id": order_id, "gpus": 1, "image": IMAGE, "ssh_keys": [pub]})
                assert placed.status == 201, placed.data
            a, b = [await wait_running(platform, o) for o in orders]
            assert a["state"] == "running" and b["state"] == "running", (a, b)

            # The tenant gets root over SSH with their key, a writable /data, a neutral hostname.
            out = ssh(tmp_path / "id", a["connection"]["port"], "id -u; hostname; touch /data/x && echo data-ok")
            assert out.returncode == 0, out.stderr
            assert out.stdout.split() == ["0", "weir", "data-ok"]
            # No password login.
            assert "passwordauthentication no" in ssh(tmp_path / "id", a["connection"]["port"], "/usr/sbin/sshd -T").stdout.lower()

            # Tenants cannot reach each other: B's address is not routable from A.
            b_name = DockerRuntime.name("dock-b")
            b_ip = docker("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", b_name)
            probe = ssh(tmp_path / "id", a["connection"]["port"], f"timeout 3 bash -c '</dev/tcp/{b_ip}/22' && echo reachable || echo blocked")
            assert probe.stdout.strip() == "blocked", probe

            # Restart keeps the data volume.
            assert (await platform.call("POST", "/v1/orders/dock-a/restart")).status == 200
            a = await wait_running(platform, "dock-a")
            assert ssh(tmp_path / "id", a["connection"]["port"], "ls /data").stdout.strip() == "x"

            # Release removes container, network and data; doing it twice is fine.
            for _ in range(2):
                assert (await platform.call("POST", "/v1/orders/dock-a/release")).data["state"] == "released"
            name = DockerRuntime.name("dock-a")
            assert docker("ps", "-a", "-q", "-f", f"name={name}") == ""
            assert docker("volume", "ls", "-q", "-f", f"name={name}") == ""
            assert docker("network", "ls", "-q", "-f", f"name={name}") == ""

            # A container that dies is reported as failed, not running.
            docker("kill", b_name)  # a manual kill is not restarted by "unless-stopped"
            await asyncio.sleep(1)
            view = (await platform.call("GET", "/v1/orders/dock-b")).data
            assert view["state"] == "failed", view
    finally:
        await server.close()
        for order_id in orders:
            name = DockerRuntime.name(order_id)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "volume", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "network", "rm", name], capture_output=True)


async def test_unknown_image_fails_the_order(tmp_path):
    runtime = DockerRuntime({IMAGE: IMAGE}, bind_ip="127.0.0.1", use_gpus=False)
    config = NodeConfig(hotkey=NODE, validators=set(), platform={PLATFORM.ss58_address},
                        gpus=gpus.simulated("RTX 4090:24564:1", "docker2"), simulated=True,
                        state_dir=tmp_path / "node", runtime=runtime, ssh_ports=(42951, 42960))
    server = TestServer(Node(config).app(), host="127.0.0.1")
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            platform = NodeClient(session, PLATFORM, f"http://127.0.0.1:{server.port}", NODE.ss58_address)
            await platform.call("POST", "/v1/orders", {"order_id": "dock-c", "gpus": 1, "image": "someone/else:latest", "ssh_keys": ["ssh-ed25519 AAAA x"]})
            view = await wait_running(platform, "dock-c", 10)
            assert view["state"] == "failed" and "image" in view["error"]
            # The GPU is free again.
            inventory = (await platform.call("GET", "/v1/inventory")).data
            assert not any(g["rented"] for g in inventory["gpus"])
    finally:
        await server.close()
