"""Signed calls from the validator to a node, with signed answers checked."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import aiohttp
import bittensor as bt
from bittensor_core import Keypair

from ..protocol import NONCE_HEADER, RESPONSE_HEADER, verify_response


class NodeError(Exception):
    """The node could not be reached, or its answer was not trustworthy."""


@dataclass
class Reply:
    status: int
    data: dict
    seconds: float


class NodeClient:
    def __init__(self, session: aiohttp.ClientSession, keypair: Keypair, base_url: str, miner_hotkey: str):
        self.session = session
        self.keypair = keypair
        self.base_url = base_url.rstrip("/")
        self.miner_hotkey = miner_hotkey

    async def call(self, method: str, path: str, payload: dict | None = None, timeout: float = 30.0) -> Reply:
        body = b"" if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        headers = bt.http_auth.sign(self.keypair, method=method, path=path, body=body, receiver_ss58=self.miner_hotkey)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        started = time.perf_counter()
        try:
            async with self.session.request(method, self.base_url + path, data=body, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                raw = await response.read()
                seconds = time.perf_counter() - started
                if not verify_response(self.miner_hotkey, headers[NONCE_HEADER], raw, response.headers.get(RESPONSE_HEADER)):
                    raise NodeError(f"unsigned or badly signed answer (HTTP {response.status})")
                try:
                    data = json.loads(raw)
                except ValueError as error:
                    raise NodeError("answer is not JSON") from error
                return Reply(response.status, data, seconds)
        except (aiohttp.ClientError, TimeoutError) as error:
            raise NodeError(f"unreachable: {type(error).__name__}") from error
