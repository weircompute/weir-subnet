import pytest

from weir_subnet.validator import codes
from weir_subnet.validator.scoring import PROBATION_SECONDS, GpuRecord, MinerRecord, NodeRecord, Policy, RentalStats, compute_weights

NOW = 1_000_000.0
IDLE_UNIT = 0.80 / 32  # one idle 4090 (weight 1.0) with target capacity 32
CAPS = {"rtx4090-24g": 1.0, "h100-80g": 1.4}


def weigh(miners, rentals, owner_uid, target_capacity, now, policy=Policy(), caps=CAPS):
    return compute_weights(miners, rentals, owner_uid, target_capacity, now, policy, caps)


def miner(uid, gpus=1, weight=1.0, price=0.5, avail=1.0, verified=True, disk=2000.0, runtime="sysbox-runc", rented=0):
    m = MinerRecord(f"hk{uid}", uid)
    m.nodes["main"] = NodeRecord("main", f"http://n{uid}", availability=avail, probes=10, price=price, code=codes.OK,
                                 disk_total_gib=disk, container_runtime=runtime)
    for i in range(gpus):
        g = GpuRecord(f"u{uid}-{i}", "rtx4090-24g", weight, NOW - 60 if verified else 0.0, code=codes.OK, index=i,
                      verified_since=NOW - PROBATION_SECONDS - 1 if verified else 0.0)
        if i < rented:
            g.code = codes.RENTED
        m.gpus[g.uuid] = g
    return m


def test_cold_start_sends_what_is_not_paid_to_the_reserve():
    weights, _ = weigh([miner(1)], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert weights[1] == pytest.approx(IDLE_UNIT)
    assert weights[0] == pytest.approx(1 - IDLE_UNIT)


def test_no_verified_supply_means_all_to_the_reserve():
    weights, _ = weigh([miner(1, verified=False)], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert weights == {0: 1.0}


def test_full_idle_supply_fills_the_idle_pool_and_unused_rented_pool_goes_to_the_reserve():
    weights, _ = weigh([miner(1, gpus=16), miner(2, gpus=16)], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert weights[1] == pytest.approx(0.40) and weights[2] == pytest.approx(0.40)
    assert weights[0] == pytest.approx(0.20)


def test_rented_pool_is_20_percent_split_by_hours():
    rentals = {"hk1": RentalStats(300.0, 1.0), "hk2": RentalStats(100.0, 1.0)}
    weights, detail = weigh([miner(1, gpus=16), miner(2, gpus=16)], rentals, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk1"]["rented"] == pytest.approx(0.15) and detail["hk2"]["rented"] == pytest.approx(0.05)
    assert 0 not in weights and sum(weights.values()) == pytest.approx(1.0)


def test_renting_to_yourself_is_worth_at_most_the_rented_pool():
    weights, _ = weigh([miner(1)], {"hk1": RentalStats(1e6, 1.0)}, owner_uid=0, target_capacity=32, now=NOW)
    assert weights[1] == pytest.approx(0.20 + IDLE_UNIT)


def test_rented_gpus_are_not_idle():
    weights, detail = weigh([miner(1, gpus=4, rented=3)], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk1"]["idle_gpus"] == 1 and weights[1] == pytest.approx(IDLE_UNIT)


def test_price_over_cap_gets_no_idle_pay():
    miners = [miner(i, price=0.5) for i in range(1, 10)] + [miner(10, price=2.0)]
    weights, detail = weigh(miners, {"hk10": RentalStats(10.0, 1.0)}, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk10"]["nodes"]["main"] == codes.PRICE_OVER_CAP
    assert detail["hk10"]["idle"] == 0 and detail["hk10"]["rented"] == pytest.approx(0.20)  # rentals still count
    assert detail["hk1"]["idle"] == pytest.approx(IDLE_UNIT, abs=1e-6)


def test_price_under_half_the_cap_gets_no_idle_pay():
    cap = CAPS["rtx4090-24g"]
    low, edge = miner(1, price=cap * 0.49), miner(2, price=cap * 0.5)
    _, detail = weigh([low, edge], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk1"]["nodes"]["main"] == codes.PRICE_UNDER_FLOOR and detail["hk1"]["idle"] == 0
    assert detail["hk2"]["nodes"]["main"] == codes.OK and detail["hk2"]["idle"] > 0


def test_disk_and_sysbox_gates():
    small, plain = miner(1, disk=10.0), miner(2, runtime="")
    _, detail = weigh([small, plain], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk1"]["nodes"]["main"] == codes.DISK_TOO_SMALL and detail["hk1"]["idle"] == 0
    assert detail["hk2"]["nodes"]["main"] == codes.NO_SYSBOX and detail["hk2"]["idle"] == 0
    _, detail = weigh([miner(2, runtime="")], {}, owner_uid=0, target_capacity=32, now=NOW, policy=Policy(require_sysbox=False))
    assert detail["hk2"]["idle"] > 0


def test_unreachable_or_flaky_node_gets_no_idle_pay():
    down, flaky = miner(1), miner(2, avail=0.3)
    down.nodes["main"].code = codes.UNREACHABLE
    _, detail = weigh([down, flaky], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk1"]["idle"] == 0 and detail["hk2"]["idle"] == 0


def test_idle_cap_per_model_shares_the_pay():
    weights, _ = weigh([miner(1, gpus=100, disk=1e5)], {}, owner_uid=0, target_capacity=1000, now=NOW, policy=Policy(idle_cap_gpus=64))
    assert weights[1] == pytest.approx(64 * 0.80 / 1000)


def test_idle_pool_never_exceeds_its_ceiling():
    weights, _ = weigh([miner(1, gpus=60, disk=1e5)], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert weights[1] == pytest.approx(0.80) and weights[0] == pytest.approx(0.20)


def test_struck_miner_gets_nothing():
    struck = miner(2, gpus=16)
    struck.struck_until = NOW + 10
    weights, _ = weigh([miner(1, gpus=16), struck], {"hk2": RentalStats(500.0, 1.0)}, owner_uid=0, target_capacity=32, now=NOW)
    assert 2 not in weights
    assert weights[1] == pytest.approx(0.40) and weights[0] == pytest.approx(0.60)


def test_model_without_a_cap_gets_no_idle_pay():
    _, detail = weigh([miner(1)], {}, owner_uid=0, target_capacity=32, now=NOW, caps={"h100-80g": 1.4})
    assert detail["hk1"]["nodes"]["main"] == codes.NO_PRICE_CAP and detail["hk1"]["idle"] == 0


def test_probation_holds_idle_pay_but_not_rentals():
    m = miner(1, gpus=2)
    for g in m.gpus.values():
        g.verified_since = NOW - PROBATION_SECONDS + 3600  # one hour short
    weights, detail = weigh([m], {"hk1": RentalStats(10.0, 1.0)}, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk1"]["idle"] == 0 and detail["hk1"]["rented"] == pytest.approx(0.20)
    weights, detail = weigh([m], {}, owner_uid=0, target_capacity=32, now=NOW + 3601)
    assert detail["hk1"]["idle"] == pytest.approx(2 * IDLE_UNIT)


def test_unpaid_activation_fee_earns_nothing():
    unpaid = miner(2, gpus=4)
    unpaid.fee_paid = False
    weights, detail = weigh([miner(1), unpaid], {"hk2": RentalStats(100.0, 1.0)}, owner_uid=0, target_capacity=32, now=NOW)
    assert 2 not in weights and detail["hk2"]["nodes"]["main"] == codes.NOT_ACTIVATED
    assert weights[1] == pytest.approx(IDLE_UNIT) and weights[0] == pytest.approx(1 - IDLE_UNIT)


def test_rest_goes_to_the_reserve_uid():
    weights, _ = weigh([miner(1)], {}, owner_uid=7, target_capacity=32, now=NOW)
    assert weights[7] == pytest.approx(1 - IDLE_UNIT) and 0 not in weights


def test_activation_fees_pay_back_faster_on_bigger_cards():
    """Payback time is fee / (weight x idle unit). The default fees are set so
    the H100 earns its fee back in two thirds of the time an RTX 4090 needs."""
    from weir_subnet.validator.models import MODELS

    by_key = {m.key: m for m in MODELS}
    payback = {k: m.fee_tao / m.weight for k, m in by_key.items()}
    assert 1.4 < payback["rtx4090-24g"] / payback["h100-80g"] < 1.6
    assert payback["h100-80g"] == min(payback.values()) and payback["rtx4090-24g"] == max(payback.values())


def test_axon_addresses_from_chain():
    from weir_subnet.validator.chain import axon_address

    assert axon_address({"ip": 1023447334, "port": 8091, "ip_type": 4}) == "61.0.145.38:8091"
    assert axon_address({"ip": 0, "port": 0, "ip_type": 0}) is None
    assert axon_address({"ip": 1, "port": 443, "ip_type": 6}) == "[::1]:443"
    assert axon_address(None) is None


def test_gpu_without_its_activation_fee_earns_no_idle_pay():
    m = miner(1, gpus=2)
    next(iter(m.gpus.values())).fee_covered = False
    weights, detail = weigh([m], {}, owner_uid=0, target_capacity=32, now=NOW)
    assert detail["hk1"]["idle"] == pytest.approx(IDLE_UNIT)
