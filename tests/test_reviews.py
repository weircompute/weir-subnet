"""Manual review: ending probation early, and disputed GPUs."""
import json

from weir_subnet.validator import codes
from weir_subnet.validator.scoring import NodeRecord, Policy, Review, compute_weights
from weir_subnet.validator.validator import _gpu_code

from test_fees import A, B, TAO, gpu, setup

NOW = 10_000_000.0


def idle_node():
    return NodeRecord("main", "http://n", availability=1.0, probes=5, price=0.3, code=codes.OK, disk_total_gib=1000, container_runtime="sysbox-runc")


def test_skip_probation_lets_a_new_gpu_earn():
    from weir_subnet.validator.scoring import MinerRecord

    g = gpu("g", "rtx4090-24g", 1.0)
    g.verified_at, g.verified_since, g.fee_covered = NOW - 60, NOW - 3600, True  # passed, but only an hour ago
    m = MinerRecord(A, 1, gpus={"g": g}, nodes={"main": idle_node()})
    caps = {"rtx4090-24g": 0.5}
    plain, _ = compute_weights([m], {}, 0, 100.0, NOW, Policy(), caps)
    assert 1 not in plain  # still in probation
    reviewed, detail = compute_weights([m], {}, 0, 100.0, NOW, Policy(), caps, {A: Review(skip_probation=True)})
    assert abs(reviewed[1] - 0.9 / 100) < 1e-9
    assert _gpu_code(g, NOW, Policy()) == codes.PROBATION and _gpu_code(g, NOW, Policy(), True) == codes.OK


def test_disputed_gpus_earn_idle_pay_without_a_node():
    from weir_subnet.validator.scoring import MinerRecord

    m = MinerRecord(B, 2)
    weights, detail = compute_weights([m], {}, 0, 100.0, NOW, Policy(), {}, {B: Review(disputed=[("h100-80g", 4.0), ("h100-80g", 4.0)])})
    assert abs(weights[2] - 2 * 4.0 * 0.9 / 100) < 1e-9
    assert detail[B]["disputed_gpus"] == 2
    struck = MinerRecord(B, 2, struck_until=NOW + 10)
    assert 2 not in compute_weights([struck], {}, 0, 100.0, NOW, Policy(), {}, {B: Review(disputed=[("h100-80g", 4.0)])})[0]


def test_disputed_gpus_pay_their_activation_fee(tmp_path):
    # 12 TAO: one H100 (10) is covered, the second is not.
    v, snap = setup(tmp_path, {"cold-b": 12 * TAO}, platform={"reviews": {B: {"disputed": {"h100-80g": 2, "not-a-model": 3}}}})
    v._apply_fees(snap)
    assert v.disputed[B] == [("h100-80g", 4.0, True), ("h100-80g", 4.0, False)]
    assert v.miners[B].fee_paid and v.credit_left["cold-b"] == 2 * TAO
    # No credit at all: nothing counts and the miner is not activated.
    v2, snap2 = setup(tmp_path, {}, platform={"reviews": {B: {"disputed": {"h100-80g": 1}}}})
    v2._apply_fees(snap2)
    assert v2.disputed[B] == [("h100-80g", 4.0, False)] and not v2.miners[B].fee_paid


def test_reviews_are_read_strictly(tmp_path):
    v, _ = setup(tmp_path, {}, platform={"reviews": {A: {"skip_probation": "yes", "disputed": {"h100-80g": True, "a100-80g": 2}}, B: "bad"}})
    assert v._reviews() == {A: {"skip_probation": False, "disputed": {"a100-80g": 2}}}


def test_failed_checks_show_as_pending_review_and_disputes_are_not_sold(tmp_path):
    v, snap = setup(tmp_path, {"cold-a": 20 * TAO}, platform={"reviews": {A: {"disputed": {"rtx4090-24g": 1}}}},
                    gpus_a=[gpu("slow", "rtx4090-24g", 1.0)])
    v.miners[A].gpus["slow"].code = codes.CHALLENGE_SLOW
    v.miners[A].nodes["main"] = idle_node()
    v.config.status_file = tmp_path / "status.json"
    v.config.sellable_file = tmp_path / "sellable.json"
    v._apply_fees(snap)
    v._write_status(snap)
    v._write_sellable(snap)
    status = json.loads((tmp_path / "status.json").read_text())["miners"][A]
    assert status["nodes"][0]["gpus"][0]["code"] == codes.PENDING_REVIEW
    assert status["nodes"][0]["gpus"][0]["reason"] == codes.CHALLENGE_SLOW
    assert status["disputed"] == [{"model": "rtx4090-24g", "count": 1, "activated": 1, "code": codes.DISPUTED}]
    assert json.loads((tmp_path / "sellable.json").read_text())["nodes"] == []
