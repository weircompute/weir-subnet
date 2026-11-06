"""Per-GPU activation fees: credit, order, grandfathering and overrides."""
import json

import pytest

from weir_subnet.validator.chain import Neuron, Snapshot
from weir_subnet.validator.scoring import GpuRecord, MinerRecord
from weir_subnet.validator.validator import Validator, ValidatorConfig

from test_end_to_end import OWNER, VALIDATOR, FakeChain, key

TAO = 10**9
A, B = key("FeeMinerA").ss58_address, key("FeeMinerB").ss58_address


def gpu(uuid, model, weight, index=0, passed=True):
    return GpuRecord(uuid, model, weight, 1.0 if passed else 0.0, "", "main", "OK", index, 1.0 if passed else 0.0, False)


def setup(tmp_path, paid_rao, platform=None, gpus_a=(), gpus_b=()):
    neurons = [Neuron(0, OWNER.ss58_address, None, True, "cold-owner"),
               Neuron(1, A, "127.0.0.1:1", False, "cold-a"), Neuron(2, B, "127.0.0.1:2", False, "cold-b")]
    path = tmp_path / "platform.json"
    path.write_text(json.dumps(platform or {}))
    v = Validator(VALIDATOR, FakeChain(neurons), ValidatorConfig(state_file=tmp_path / "s.json", platform_file=path, fee_address="fee"))
    v.fees_paid = dict(paid_rao)
    v.miners[A] = MinerRecord(A, 1, gpus={g.uuid: g for g in gpus_a})
    v.miners[B] = MinerRecord(B, 2, gpus={g.uuid: g for g in gpus_b})
    return v, Snapshot(100, OWNER.ss58_address, neurons)


def test_each_gpu_pays_most_valuable_first(tmp_path):
    # 12 TAO covers the H100 (10) but then not a 4090 (3.7); the rest stays as credit.
    v, snap = setup(tmp_path, {"cold-a": 12 * TAO},
                    gpus_a=[gpu("g4090", "rtx4090-24g", 1.0, 0), gpu("gh100", "h100-80g", 4.0, 1)])
    v._apply_fees(snap)
    gpus = v.miners[A].gpus
    assert gpus["gh100"].fee_covered and not gpus["g4090"].fee_covered
    assert v.miners[A].fee_paid and v.credit_left["cold-a"] == 2 * TAO


def test_unverified_gpus_and_short_credit(tmp_path):
    v, snap = setup(tmp_path, {"cold-b": 3 * TAO}, gpus_b=[gpu("x", "rtx4090-24g", 1.0)],
                    gpus_a=[gpu("y", "rtx4090-24g", 1.0, passed=False)], platform={})
    v.fees_paid["cold-a"] = 10 * TAO
    v._apply_fees(snap)
    assert not v.miners[B].fee_paid and not v.miners[B].gpus["x"].fee_covered
    assert not v.miners[A].gpus["y"].fee_covered and v.miners[A].fee_paid  # credit waits until the GPU passes
    assert v.credit_left["cold-a"] == 10 * TAO


def test_covered_gpus_stay_covered_after_a_fee_change(tmp_path):
    v, snap = setup(tmp_path, {"cold-a": 4 * TAO}, gpus_a=[gpu("g", "rtx4090-24g", 1.0)])
    v._apply_fees(snap)
    assert v.miners[A].gpus["g"].fee_covered
    v._save()
    (tmp_path / "platform.json").write_text(json.dumps({"fees": {"rtx4090-24g": 20}}))
    again = Validator(VALIDATOR, v.chain, v.config)
    again.miners[A] = MinerRecord(A, 1, gpus={"g": gpu("g", "rtx4090-24g", 1.0), "h": gpu("h", "rtx4090-24g", 1.0, 1)})
    again._apply_fees(snap)
    assert again.miners[A].gpus["g"].fee_covered and not again.miners[A].gpus["h"].fee_covered


def test_platform_credit_and_weight_override(tmp_path):
    v, snap = setup(tmp_path, {}, platform={"activation_credit": {B: 7.4}, "weights": {"rtx4090-24g": 0.5}, "fees": {"nope": 1}},
                    gpus_b=[gpu("p", "rtx4090-24g", 1.0), gpu("q", "rtx4090-24g", 1.0, 1)])
    v._apply_fees(snap)
    assert all(g.fee_covered for g in v.miners[B].gpus.values())
    assert v._per_model("weights")["rtx4090-24g"] == 0.5 and "nope" not in v._per_model("fees")


def test_no_fee_address_covers_everything(tmp_path):
    v, snap = setup(tmp_path, {}, gpus_a=[gpu("g", "h100-80g", 4.0)])
    v.config.fee_address = ""
    v._apply_fees(snap)
    assert v.miners[A].fee_paid and v.miners[A].gpus["g"].fee_covered


def test_gpu_on_another_coldkey_is_not_covered(tmp_path):
    v, snap = setup(tmp_path, {}, gpus_b=[gpu("g", "h100-80g", 4.0)])
    v.covered["g"] = {"hotkey": A, "coldkey": "cold-a", "rao": 10 * TAO}
    v._apply_fees(snap)
    assert not v.miners[B].gpus["g"].fee_covered



def test_older_activations_without_a_model_are_not_rechecked(tmp_path):
    v, snap = setup(tmp_path, {"cold-a": 3701 * TAO // 1000}, gpus_a=[gpu("g", "h100-80g", 4.0)])
    v.covered["g"] = {"hotkey": A, "coldkey": "cold-a", "rao": 37 * TAO // 10}
    v._apply_fees(snap, now=2000.0)
    g = v.miners[A].gpus["g"]
    assert g.fee_covered and g.fee_short_rao == 0


def test_new_activations_record_the_model_and_switching_up_owes(tmp_path):
    v, snap = setup(tmp_path, {"cold-a": 3701 * TAO // 1000}, gpus_a=[gpu("g", "rtx4090-24g", 1.0)])
    v._apply_fees(snap, now=1000.0)
    assert v.covered["g"]["model"] == "rtx4090-24g"
    g = v.miners[A].gpus["g"]
    g.model, g.weight = "h100-80g", 4.0
    v._apply_fees(snap, now=2000.0)
    assert not g.fee_covered and g.fee_short_rao == 63 * TAO // 10


def test_a_marked_gpu_on_a_dearer_model_owes_the_difference(tmp_path):
    from weir_subnet.validator.validator import TOP_UP_SECONDS

    v, snap = setup(tmp_path, {"cold-a": 3701 * TAO // 1000}, gpus_a=[gpu("g", "h100-80g", 4.0)])
    v.covered["g"] = {"hotkey": A, "coldkey": "cold-a", "rao": 37 * TAO // 10, "model": "rtx4090-24g"}
    g = v.miners[A].gpus["g"]
    v._apply_fees(snap, now=2000.0)
    assert not g.fee_covered and not v.miners[A].fee_paid
    assert g.fee_short_rao == 63 * TAO // 10 and g.fee_due == 2000.0 + TOP_UP_SECONDS
    v._apply_fees(snap, now=2000.0 + 86400)
    assert g.fee_due == 2000.0 + TOP_UP_SECONDS  # the deadline does not move
    v.fees_paid["cold-a"] += 63 * TAO // 10
    v._apply_fees(snap, now=3000.0)
    assert g.fee_covered and g.fee_short_rao == 0 and v.covered["g"] == {"hotkey": A, "coldkey": "cold-a", "rao": 10 * TAO, "model": "h100-80g"}
    assert v.miners[A].fee_paid and v.credit_left["cold-a"] == TAO // 1000


def test_an_unpaid_difference_drops_the_activation_after_a_week(tmp_path):
    from weir_subnet.validator.validator import TOP_UP_SECONDS

    v, snap = setup(tmp_path, {"cold-a": 3701 * TAO // 1000}, gpus_a=[gpu("g", "h100-80g", 4.0)])
    v.covered["g"] = {"hotkey": A, "coldkey": "cold-a", "rao": 37 * TAO // 10, "model": "rtx4090-24g"}
    g = v.miners[A].gpus["g"]
    v._apply_fees(snap, now=0.0)
    assert not g.fee_covered and g.fee_short_rao
    v._apply_fees(snap, now=TOP_UP_SECONDS)
    assert "g" not in v.covered and not g.fee_covered and g.verified_since == 0.0
    assert v.credit_left["cold-a"] == 3701 * TAO // 1000  # what was paid is credit again


def test_same_or_cheaper_model_needs_nothing_more(tmp_path):
    v, snap = setup(tmp_path, {"cold-a": 10 * TAO}, gpus_a=[gpu("g", "h100-80g", 4.0)])
    v._apply_fees(snap, now=0.0)
    g = v.miners[A].gpus["g"]
    g.model, g.weight = "rtx4090-24g", 1.0
    v._apply_fees(snap, now=10.0)
    assert g.fee_covered and g.fee_short_rao == 0
