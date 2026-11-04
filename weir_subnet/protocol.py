"""Wire rules shared by the node and the validator.

Requests use the standard Bittensor signed-request format (btauth/1): the
caller signs with its hotkey and names the node's hotkey as receiver.

Responses are signed too, so a validator knows an answer came from the miner
it asked and not from something in between. The node signs
    weir-resp/1 \\n <request nonce> \\n <sha256 of the response body>
with its hotkey and returns the signature in X-Weir-Signature.
"""

from __future__ import annotations

import hashlib

from bittensor_core import Keypair

AGENT_VERSION = "0.1.0"
RESPONSE_HEADER = "X-Weir-Signature"
NONCE_HEADER = "X-Bittensor-Nonce"


def response_payload(request_nonce: str, body: bytes) -> bytes:
    return f"weir-resp/1\n{request_nonce}\n{hashlib.sha256(body).hexdigest()}".encode()


def sign_response(keypair: Keypair, request_nonce: str, body: bytes) -> str:
    return "0x" + keypair.sign(response_payload(request_nonce, body)).hex()


def verify_response(hotkey_ss58: str, request_nonce: str, body: bytes, signature: str | None) -> bool:
    if not signature or not signature.startswith("0x"):
        return False
    try:
        return bool(Keypair(ss58_address=hotkey_ss58).verify(response_payload(request_nonce, body), bytes.fromhex(signature[2:])))
    except Exception:
        return False
