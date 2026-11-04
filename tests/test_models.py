"""Time limits against real measurements (A800 SXM4 40GB, 2026-10-30)."""

from weir_subnet import challenge
from weir_subnet.validator.models import classify, time_limit

# selftest --json on the card: (n, pairs, seconds)
MEASURED_A800 = [(8192, 112, 6.63), (8192, 112, 6.735), (8192, 112, 6.813), (8192, 112, 7.1), (7168, 146, 6.792), (7168, 146, 6.785)]


def test_a800_passes_with_margin_and_half_speed_fails():
    model = classify("A800-SXM4-40GB", 40960)
    assert model and model.calibrated
    for n, pairs, seconds in MEASURED_A800:
        spec = challenge.Spec("x", 1, n, pairs)
        assert pairs == challenge.size_for(40960, n)
        limit = time_limit(spec.flops, spec.elements, model)
        assert seconds * 1.2 < limit  # at least 20% headroom for a real card
        assert seconds * 2 > limit  # a card half as fast fails


def test_every_model_gets_a_limit_for_every_size():
    for vram, name in ((24564, "RTX 4090"), (81559, "H100 80GB HBM3"), (32607, "NVIDIA GeForce RTX 5090")):
        model = classify(name, vram)
        for n in (6144, 7168, 8192):
            spec = challenge.Spec("x", 1, n, challenge.size_for(vram, n))
            assert 3 < time_limit(spec.flops, spec.elements, model) < 60
