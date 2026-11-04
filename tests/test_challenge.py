import random

import numpy as np
import pytest

from weir_subnet import challenge
from weir_subnet.challenge import Spec


def spec(n=128, count=3):
    return Spec("t", random.getrandbits(62), n, count)


def test_gpu_and_cpu_solvers_agree():
    torch = pytest.importorskip("torch")
    s = spec()
    assert np.array_equal(challenge.solve_numpy(s), challenge.solve_torch(s, "cpu"))
    if torch.cuda.is_available():
        assert np.array_equal(challenge.solve_numpy(s), challenge.solve_torch(s, "cuda:0"))


def test_honest_answer_verifies():
    s = spec()
    fps = challenge.solve_numpy(s)
    picks = [(0, 1), (2, 127), (2, 3)]
    verdict = challenge.verify(s, challenge.commitment(s, fps), fps.ravel().tolist(), picks, challenge.reveal_rows(s, picks))
    assert verdict.ok, verdict.reason


def test_partial_work_is_caught():
    """A node that skipped one matrix and guessed its fingerprints."""
    s = spec(count=4)
    fps = challenge.solve_numpy(s)
    fps[3] = np.random.randint(-(2**62), 2**62, size=s.n)
    root = challenge.commitment(s, fps)
    picks = [(3, 10)]
    # The real row does not match the guessed fingerprint it committed to.
    verdict = challenge.verify(s, root, fps.ravel().tolist(), picks, challenge.reveal_rows(s, picks))
    assert not verdict.ok and "fingerprint" in verdict.reason


def test_changed_fingerprints_break_the_commitment():
    s = spec()
    fps = challenge.solve_numpy(s)
    root = challenge.commitment(s, fps)
    fps[1, 5] ^= 1
    assert not challenge.verify(s, root, fps.ravel().tolist(), [(0, 0)], challenge.reveal_rows(s, [(0, 0)])).ok


def test_wrong_row_is_caught():
    s = spec()
    fps = challenge.solve_numpy(s)
    rows = challenge.reveal_rows(s, [(1, 9)])
    rows[0][0] += 1
    assert not challenge.verify(s, challenge.commitment(s, fps), fps.ravel().tolist(), [(1, 9)], rows).ok


def test_different_seeds_give_different_work():
    a, b = Spec("x", 1, 64, 1), Spec("x", 2, 64, 1)
    assert not np.array_equal(challenge.rows_np(a, 0, 0, 0, 64), challenge.rows_np(b, 0, 0, 0, 64))
    # A and B of one pair, and different pairs, never share elements
    s = Spec("x", 7, 64, 2)
    blocks = [challenge.rows_np(s, i, salt, 0, 64) for i in range(2) for salt in (0, 1)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(blocks[i], blocks[j])


def test_products_stay_exact_in_fp16():
    s = Spec("x", 99, 8192, 1)
    row = challenge.product_rows(s, 0, [0, 4000])
    assert np.abs(row).max() < 2048  # far inside fp16's exact integer range


def test_spec_bounds():
    with pytest.raises(ValueError):
        Spec.from_json({"protocol": "weir-mm/1", "challenge_id": "x", "seed": 1, "n": 100, "count": 1})
    with pytest.raises(ValueError):
        Spec.from_json({"protocol": "other", "challenge_id": "x", "seed": 1, "n": 128, "count": 1})


def test_size_fills_memory():
    # A and B for one pair at n = 8192 take 256 MiB in fp16.
    assert challenge.size_for(24564) == 67  # 24 GB card, 70% fill -> about 17 GB
    assert challenge.size_for(81559) == 223


def test_checked_rows_spread_over_matrices():
    rng = random.Random(1)
    s = Spec("t", 1, 128, 40)
    for _ in range(50):
        picks = challenge.pick_rows(s, 4, rng)
        assert len({b for b, _ in picks}) == 4  # four different matrix pairs
    assert len(challenge.pick_rows(Spec("t", 1, 128, 2), 4, rng)) >= 2


def test_skipping_a_quarter_is_caught_most_of_the_time():
    """A node that guesses the last quarter of the pairs."""
    s = Spec("t", 5, 64, 8)
    honest = challenge.solve_numpy(s)
    cheat = honest.copy()
    cheat[6:] = np.random.default_rng(0).integers(-(2**62), 2**62, size=(2, s.n))
    root = challenge.commitment(s, cheat)
    rng = random.Random(7)
    caught = 0
    for _ in range(200):
        picks = challenge.pick_rows(s, 4, rng)
        caught += not challenge.verify(s, root, cheat.ravel().tolist(), picks, challenge.reveal_rows(s, picks)).ok
    assert caught / 200 > 0.7  # 1 - C(6,4)/C(8,4) = 79% with distinct pairs


def test_selftest_runs_the_validator_checks():
    import asyncio

    from weir_subnet.node import gpus, selftest

    checks = asyncio.run(selftest.run(gpus.simulated("RTX 4090:24564:2", "t") + gpus.simulated("Tesla T4:15360:1", "u"), n=128, simulated=True))
    by_name = {c.name: c for c in checks}
    assert by_name["GPU 0 RTX 4090"].result == "PASS" and by_name["GPU 0 RTX 4090"].measured["model"] == "rtx4090-24g"
    assert by_name["GPU 0 Tesla T4"].result == "FAIL"
    assert by_name["Payout address"].result == "WARN"
