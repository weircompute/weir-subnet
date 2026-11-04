"""Where tenant workloads run on a miner's box.

DockerRuntime gives each order:
- its own container, named by a hash of the order id (no tenant data on the host);
- its own bridge network, so tenants on one box cannot reach each other;
- its own data volume at /data, deleted on release;
- the ordered GPUs only, a CPU and memory share, and SSH on one host port.

With Sysbox installed (`--docker-runtime sysbox-runc`) tenants get root and
can run Docker inside without the container being privileged. Without it,
the container is an ordinary one: root inside, but no Docker-in-Docker.

All calls are idempotent: launching an order that already has a container,
or releasing one that is already gone, does nothing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from dataclasses import dataclass, field
from typing import Protocol

# Starts sshd with the tenant's keys only. Installs openssh-server on first
# boot if the image lacks it (Debian/Ubuntu images).
BOOT = r"""
set -e
umask 077
mkdir -p /root/.ssh
printf '%s' "$WEIR_SSH_KEYS" | base64 -d > /root/.ssh/authorized_keys
unset WEIR_SSH_KEYS
if ! command -v sshd >/dev/null 2>&1 && [ ! -x /usr/sbin/sshd ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq --no-install-recommends openssh-server >/dev/null
fi
mkdir -p /run/sshd /etc/ssh/sshd_config.d
# Kept in a config file too, so a restart of sshd by the tenant keeps them.
printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin prohibit-password\n' > /etc/ssh/sshd_config.d/00-weir.conf
grep -q '^Include /etc/ssh/sshd_config.d' /etc/ssh/sshd_config || sed -i '1i Include /etc/ssh/sshd_config.d/*.conf' /etc/ssh/sshd_config
ssh-keygen -A >/dev/null
exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password -o KbdInteractiveAuthentication=no
"""


class Runtime(Protocol):
    async def launch(self, order) -> None: ...
    async def status(self, order) -> str: ...  # starting | running | failed | gone
    async def restart(self, order) -> None: ...
    async def release(self, order) -> None: ...


class StateOnlyRuntime:
    """Development stand-in: tracks orders without starting anything."""

    async def launch(self, order) -> None:
        return None

    async def status(self, order) -> str:
        return "running"

    async def restart(self, order) -> None:
        return None

    async def release(self, order) -> None:
        return None


class RuntimeFailure(Exception):
    pass


@dataclass
class DockerRuntime:
    images: dict[str, str]  # Weir image id -> local Docker image
    bind_ip: str = "0.0.0.0"
    docker_runtime: str = ""  # e.g. "sysbox-runc"
    use_gpus: bool = True
    cpus_per_gpu: float = 8.0
    memory_gib_per_gpu: int = 48
    docker: str = "docker"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @staticmethod
    def name(order_id: str) -> str:
        return "weir-order-" + hashlib.sha256(order_id.encode()).hexdigest()[:16]

    async def _run(self, *args: str, check: bool = True) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(self.docker, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        text = out.decode(errors="replace").strip()
        if check and proc.returncode != 0:
            raise RuntimeFailure(f"docker {args[0]} failed: {text[-300:]}")
        return proc.returncode, text

    async def _exists(self, kind: str, name: str) -> bool:
        code, _ = await self._run(kind, "inspect", name, check=False)
        return code == 0

    async def launch(self, order) -> None:
        image = self.images.get(order.image)
        if not image:
            raise RuntimeFailure("image not offered on this node")
        name = self.name(order.order_id)
        async with self._lock:
            if await self._exists("container", name):
                return
            if not await self._exists("network", name):
                await self._run("network", "create", "--driver", "bridge", "--label", "weir=order", name)
            if not await self._exists("volume", name):
                await self._run("volume", "create", "--label", "weir=order", name)
            gpus = max(1, len(order.gpus))
            args = [
                "run", "-d", "--name", name, "--hostname", "weir", "--label", "weir=order",
                "--network", name, "--restart", "unless-stopped",
                "-p", f"{self.bind_ip}:{order.port}:22",
                "-v", f"{name}:/data",
                "--cpus", str(self.cpus_per_gpu * gpus), "--memory", f"{self.memory_gib_per_gpu * gpus}g",
                "--pids-limit", "8192", "--security-opt", "no-new-privileges",
                "-e", "WEIR_SSH_KEYS=" + base64.b64encode("\n".join(order.ssh_keys).encode()).decode(),
            ]
            if self.use_gpus:
                args += ["--gpus", '"device=' + ",".join(str(i) for i in order.gpus) + '"']
            if self.docker_runtime:
                args += ["--runtime", self.docker_runtime]
            args += ["--entrypoint", "/bin/sh", image, "-c", BOOT]
            await self._run(*args)

    async def status(self, order) -> str:
        code, state = await self._run("inspect", "-f", "{{.State.Status}}", self.name(order.order_id), check=False)
        if code != 0:
            return "gone"
        if state in ("exited", "dead"):
            return "failed"
        if state != "running":
            return "starting"
        return "running" if await self._ssh_ready(order.port) else "starting"

    async def _ssh_ready(self, port: int) -> bool:
        host = "127.0.0.1" if self.bind_ip in ("0.0.0.0", "") else self.bind_ip
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 3)
            banner = await asyncio.wait_for(reader.readline(), 3)
            writer.close()
            return banner.startswith(b"SSH-")
        except (OSError, asyncio.TimeoutError):
            return False

    async def restart(self, order) -> None:
        await self._run("restart", self.name(order.order_id))

    async def release(self, order) -> None:
        name = self.name(order.order_id)
        async with self._lock:
            await self._run("rm", "-f", "-v", name, check=False)
            await self._run("volume", "rm", "-f", name, check=False)  # tenant data goes with the order
            await self._run("network", "rm", name, check=False)
            if await self._exists("container", name) or await self._exists("volume", name):
                raise RuntimeFailure("release did not complete")
