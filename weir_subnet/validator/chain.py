"""The validator's view of the chain, behind a small interface so the loop
can be tested without a network."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Protocol


def axon_address(axon) -> str | None:
    """The chain stores an axon as {"ip": int, "port": int, "ip_type": 4|6, ...}.
    Returns "host:port", or None when nothing is published."""
    if not axon:
        return None
    if isinstance(axon, str):
        return axon
    ip, port = int(axon.get("ip", 0)), int(axon.get("port", 0))
    if not ip or not port:
        return None
    try:
        if int(axon.get("ip_type", 4)) == 6:
            return f"[{ipaddress.IPv6Address(ip)}]:{port}"
        return f"{ipaddress.IPv4Address(ip)}:{port}"
    except ValueError:
        return None


@dataclass
class Neuron:
    uid: int
    hotkey: str
    axon: str | None  # "ip:port"
    validator_permit: bool
    coldkey: str = ""


# Share of a subnet's alpha emission that goes to miners (Yuma: 41%).
MINER_SHARE = 0.41


@dataclass
class Snapshot:
    block: int
    owner_hotkey: str
    neurons: list[Neuron]
    alpha_tao: float = 0.0  # pool price: TAO per alpha
    miner_alpha_per_tempo: float = 0.0  # alpha paid to miners each tempo

    @property
    def owner_uid(self) -> int:
        for n in self.neurons:
            if n.hotkey == self.owner_hotkey:
                return n.uid
        return 0


class Chain(Protocol):
    def snapshot(self) -> Snapshot: ...
    def set_weights(self, weights: dict[int, float]) -> None: ...
    def transfers_to(self, address: str, start: int, end: int) -> list[tuple[str, int, int]]: ...


class BittensorChain:
    """Reads the metagraph and sets weights with the validator hotkey."""

    def __init__(self, network: str, netuid: int, wallet_name: str, hotkey: str, wallet_path: str, version_key: int = 0):
        import bittensor as bt

        self.bt = bt
        self.network, self.netuid, self.version_key = network, netuid, version_key
        self.wallet = bt.Wallet(name=wallet_name, hotkey=hotkey, path=wallet_path)
        self.subtensor = bt.Subtensor(network)

    def snapshot(self) -> Snapshot:
        raw = self.subtensor.read("metagraph", netuid=self.netuid)
        neurons = [
            Neuron(uid, hotkey, axon_address(raw["axons"][uid]), bool(raw["validator_permit"][uid]), raw["coldkeys"][uid])
            for uid, hotkey in enumerate(raw["hotkeys"])
        ]
        alpha_in = int(raw.get("alpha_in") or 0)
        alpha_tao = int(raw.get("tao_in") or 0) / alpha_in if alpha_in else 0.0
        miner_alpha = int(raw.get("alpha_out_emission") or 0) / 1e9 * int(raw.get("tempo") or 0) * MINER_SHARE
        return Snapshot(int(raw["block"]), raw["owner_hotkey"], neurons, alpha_tao, miner_alpha)

    def set_weights(self, weights: dict[int, float]) -> None:
        # Picks plain or commit-reveal on its own, and raises on failure.
        self.bt.set_weights(self.netuid, weights, wallet=self.wallet, network=self.network, version_key=self.version_key)

    def transfers_to(self, address: str, start: int, end: int) -> list[tuple[str, int, int]]:
        """(sender, rao, block) for every TAO transfer to `address` in blocks
        start..end inclusive. Older blocks need an archive endpoint."""
        substrate = self.subtensor._client._substrate
        found = []
        for block in range(start, end + 1):
            block_hash = self.subtensor._call(substrate.block_hash(block))
            for record in self.subtensor._call(substrate.events(block_hash)):
                event = record.get("event", record)
                if event.get("module_id") == "Balances" and event.get("event_id") == "Transfer":
                    attrs = event.get("attributes") or {}
                    if attrs.get("to") == address:
                        found.append((str(attrs.get("from")), int(attrs.get("amount", 0)), block))
        return found
