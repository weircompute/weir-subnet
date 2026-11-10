"""Reason codes the validator records for every node and GPU.

Miners see them in their status (`weir-node status`), so each one says what
happened and what to do about it.
"""

OK = "OK"  # passed recently
PENDING = "PENDING"  # seen, not challenged yet
RENTED = "RENTED"  # in use by a tenant; keeps its last pass
BUSY = "BUSY"  # a challenge was already running on it

UNREACHABLE = "UNREACHABLE"  # no connection or timeout: check firewall, port, process
BAD_SIGNATURE = "BAD_SIGNATURE"  # answer not signed by the registered hotkey: wrong hotkey on the node
HTTP_ERROR = "HTTP_ERROR"  # node answered with an error
FLEET_INVALID = "FLEET_INVALID"  # fleet entry does not match the node's own name, or too many nodes

UNKNOWN_MODEL = "UNKNOWN_MODEL"  # GPU name and memory match no accepted model
BAD_GPU_ID = "BAD_GPU_ID"  # the GPU's UUID is not one an NVIDIA driver gives: not counted
GPU_SET_CHANGED = "GPU_SET_CHANGED"  # GPUs differ from last time: every GPU is checked again
DUPLICATE_GPU = "DUPLICATE_GPU"  # this GPU is already registered under another hotkey or node

CHALLENGE_REFUSED = "CHALLENGE_REFUSED"  # node did not take the challenge
CHALLENGE_SLOW = "CHALLENGE_SLOW"  # correct but slower than the model allows
CHALLENGE_WRONG = "CHALLENGE_WRONG"  # wrong answer: miner scores 0 for 24 hours
STRUCK = "STRUCK"  # miner is in its 24-hour zero period

PRICE_OVER_CAP = "PRICE_OVER_CAP"  # price above the model's cap: no idle pay, not sold
PRICE_UNDER_FLOOR = "PRICE_UNDER_FLOOR"  # price below half the model's cap: no idle pay, not sold
NO_PRICE_CAP = "NO_PRICE_CAP"  # no cap published for this model yet: no idle pay, not sold
PROBATION = "PROBATION"  # verified, but not yet for the full waiting period
NOT_ACTIVATED = "NOT_ACTIVATED"  # activation fee not paid (for the hotkey, or for this GPU): no idle pay, not sold
DISK_TOO_SMALL = "DISK_TOO_SMALL"  # idle pay needs total disk >= 1.5x total VRAM
NO_SYSBOX = "NO_SYSBOX"  # idle pay needs tenants to run under Sysbox
PENDING_REVIEW = "PENDING_REVIEW"  # did not pass, or not an accepted model: no pay unless a review decides otherwise
DISPUTED = "DISPUTED"  # check result disputed and under review: counted for idle pay, not sold
FEE_SHORT = "FEE_SHORT"  # activated as a cheaper model than it reports now: no pay, not sold until the difference is paid

# Shown as PENDING_REVIEW (the original code stays in "reason").
NEEDS_REVIEW = (UNKNOWN_MODEL, CHALLENGE_REFUSED, CHALLENGE_SLOW, CHALLENGE_WRONG)


EXPLAIN = {
    OK: "passed recently; this GPU is being sold",
    PENDING: "seen; waiting for its first check",
    RENTED: "rented by a tenant; keeps its last pass",
    BUSY: "a check was already running; tried again next round",
    UNREACHABLE: "could not connect: check the process, port and firewall",
    BAD_SIGNATURE: "answers are not signed by your registered hotkey: check --wallet/--hotkey on this machine",
    HTTP_ERROR: "the node answered with an error: check its log",
    FLEET_INVALID: "the fleet entry does not match this machine's --name, or the fleet is too large",
    UNKNOWN_MODEL: "this GPU is not on the accepted list",
    BAD_GPU_ID: "the GPU's UUID is not what nvidia-smi reports for a real card: run the unmodified weir-node",
    GPU_SET_CHANGED: "GPUs changed since last time: every GPU is checked again",
    DUPLICATE_GPU: "this GPU is already registered elsewhere; only the first registration counts",
    CHALLENGE_REFUSED: "the node did not take the check: check its log",
    CHALLENGE_SLOW: "correct, but slower than this model must be: check power limits, clocks, other load",
    CHALLENGE_WRONG: "wrong answer: your score is 0 for 24 hours",
    STRUCK: "inside the 24-hour zero period after a wrong answer",
    PRICE_OVER_CAP: "your price is above this model's cap: idle GPUs earn no idle pay and are not sold",
    PRICE_UNDER_FLOOR: "your price is below half of this model's cap: idle GPUs earn no idle pay and are not sold",
    NO_PRICE_CAP: "no price cap is published for this model yet: idle GPUs earn no idle pay and are not sold",
    PROBATION: "verified; idle pay starts once it has passed checks without a gap for the waiting period",
    NOT_ACTIVATED: "no activation fee received for this GPU (or for any GPU of this hotkey): it earns no idle pay and is not sold",
    DISK_TOO_SMALL: "total disk is under 1.5x total VRAM: idle GPUs earn no idle pay",
    NO_SYSBOX: "tenants do not run under Sysbox (--docker-runtime sysbox-runc): idle GPUs earn no idle pay",
    PENDING_REVIEW: "did not pass the check, or is not an accepted model: it earns nothing; contact Weir for a review",
    DISPUTED: "the check result is under review: counted for idle pay meanwhile, not listed for rent",
    FEE_SHORT: ("this GPU was activated as a cheaper model than it reports now: it earns nothing and is not sold until the difference "
                "is paid; if it is not paid in time, the activation is dropped and its waiting period starts again"),
}
