"""tests/test_metrics.py — Unit tests for evaluation metrics."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, hit_rate_at_k, mrr, catalog_coverage

class TestNDCG:
    def test_perfect(self):   assert ndcg_at_k(["a","b","c"], {"a","b","c"}, 3) == pytest.approx(1.0)
    def test_no_hits(self):   assert ndcg_at_k(["x","y"], {"a","b"}, 2) == 0.0
    def test_position(self):
        assert ndcg_at_k(["a","x","x"], {"a"}, 3) > ndcg_at_k(["x","x","a"], {"a"}, 3)
    def test_empty_rel(self): assert ndcg_at_k(["a","b"], set(), 2) == 0.0
    def test_k_cutoff(self):
        assert ndcg_at_k(["x","x","x","a"], {"a"}, 3) == 0.0
        assert ndcg_at_k(["x","x","x","a"], {"a"}, 4) > 0.0

class TestPrecisionRecall:
    def test_precision_perfect(self): assert precision_at_k(["a","b"], {"a","b"}, 2) == pytest.approx(1.0)
    def test_precision_zero(self):    assert precision_at_k(["x","y"], {"a","b"}, 2) == 0.0
    def test_precision_half(self):    assert precision_at_k(["a","x"], {"a","b"}, 2) == pytest.approx(0.5)
    def test_recall_perfect(self):    assert recall_at_k(["a","b"], {"a","b"}, 2) == pytest.approx(1.0)
    def test_recall_partial(self):    assert recall_at_k(["a","x"], {"a","b","c"}, 2) == pytest.approx(1/3)

class TestHitRate:
    def test_hit(self):    assert hit_rate_at_k(["a","b"], {"a"}, 2) == 1.0
    def test_no_hit(self): assert hit_rate_at_k(["x","y"], {"a"}, 2) == 0.0

class TestMRR:
    def test_rank1(self): assert mrr(["a","b"], {"a"}) == pytest.approx(1.0)
    def test_rank2(self): assert mrr(["x","a"], {"a"}) == pytest.approx(0.5)
    def test_miss(self):  assert mrr(["x","y"], {"a"}) == 0.0

class TestCoverage:
    def test_full(self):    assert catalog_coverage([["a","b"],["c","d"]], 4) == 1.0
    def test_partial(self): assert catalog_coverage([["a","b"],["a","b"]], 4) == pytest.approx(0.5)
    def test_empty(self):   assert catalog_coverage([], 0) == 0.0
