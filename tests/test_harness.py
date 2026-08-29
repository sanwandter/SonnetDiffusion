import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision2sonnet"))
sys.path.insert(0, os.path.join(ROOT, "resultados", "experimentos"))

import harness


# --- a synthetic sonnet that satisfies ABBA ABBA CDE CDE exactly -------------
# groups: A=[1,4,5,8] B=[2,3,6,7] C=[9,12] D=[10,13] E=[11,14]
_ENDINGS = ["amor", "vida", "vida", "amor",
            "amor", "vida", "vida", "amor",
            "mar", "sol", "luz", "mar", "sol", "luz"]
PERFECT = " ".join(f"canta el {w} <eos>" for w in _ENDINGS)


def test_perfect_sonnet_scores_full_format_and_rhyme():
    m = harness.score_lines([PERFECT])
    assert m["format_pct"] == 100.0
    assert m["rhyme_full_pct"] == 100.0


def test_score_lines_strips_examples_key():
    m = harness.score_lines([PERFECT])
    assert "_examples" not in m


def test_thirteen_verses_fails_format():
    broken = " ".join(f"canta el {w} <eos>" for w in _ENDINGS[:13])
    m = harness.score_lines([broken])
    assert m["format_pct"] == 0.0


def test_score_one_returns_per_sonnet_rhyme():
    s = harness.score_one(PERFECT)
    assert s["n_lines"] == 14
    assert s["rhyme"] == pytest.approx(1.0)


def test_mean_ci_known_values():
    # values 1..5: mean 3.0, sample sd 1.5811388, t(0.975, df=4) = 2.776445
    # halfwidth = 2.776445 * 1.5811388 / sqrt(5) = 1.963243
    r = harness.mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert r["mean"] == pytest.approx(3.0)
    assert r["halfwidth"] == pytest.approx(1.963243, abs=1e-5)
    assert r["lo"] == pytest.approx(3.0 - 1.963243, abs=1e-5)
    assert r["hi"] == pytest.approx(3.0 + 1.963243, abs=1e-5)
    assert r["n"] == 5


def test_mean_ci_single_value_has_zero_halfwidth():
    r = harness.mean_ci([7.0])
    assert r["mean"] == pytest.approx(7.0)
    assert r["halfwidth"] == 0.0
    assert r["n"] == 1


def test_mean_ci_rejects_empty():
    with pytest.raises(ValueError):
        harness.mean_ci([])


def test_aggregate_seeds_produces_ci_per_key():
    per_seed = [
        {"format_pct": 100.0, "rhyme_full_pct": 90.0},
        {"format_pct": 100.0, "rhyme_full_pct": 94.0},
        {"format_pct": 100.0, "rhyme_full_pct": 92.0},
    ]
    agg = harness.aggregate_seeds(per_seed, ["format_pct", "rhyme_full_pct"])
    assert agg["format_pct"]["mean"] == pytest.approx(100.0)
    assert agg["format_pct"]["halfwidth"] == pytest.approx(0.0)
    assert agg["rhyme_full_pct"]["mean"] == pytest.approx(92.0)
    assert agg["rhyme_full_pct"]["halfwidth"] > 0.0
    assert agg["rhyme_full_pct"]["n"] == 3


def test_set_seed_makes_torch_reproducible():
    harness.set_seed(102)
    a = torch.randn(8)
    harness.set_seed(102)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_differ():
    harness.set_seed(102)
    a = torch.randn(8)
    harness.set_seed(202)
    b = torch.randn(8)
    assert not torch.equal(a, b)
