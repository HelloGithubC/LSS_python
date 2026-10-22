"""Contracts for the separation of results, counting and resampling."""

import importlib
import subprocess
import sys

import numpy as np
import pytest


def test_results_import_does_not_load_counting_or_compressors():
    program = """
import builtins
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.startswith('Corrfunc'):
        raise AssertionError('Results imported Corrfunc')
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from LSS_python.xismu import xismu
import sys
assert xismu.__module__ == 'LSS_python.xismu'
for name in ('LSS_python.tpcf', 'LSS_python.cov', 'LSS_python.compressor'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=60)


def test_old_class_lookup_warns_once_and_returns_canonical_class():
    program = """
import warnings
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter('always', FutureWarning)
    from LSS_python.xismu import xismu
    import LSS_python.tpcf as tpcf
    assert not seen
    assert tpcf.xismu is xismu
    from LSS_python.tpcf import xismu as old_class
    assert old_class is xismu
    assert tpcf.xismu is xismu
    assert len(seen) == 1, seen
    assert seen[0].category is FutureWarning
    text = str(seen[0].message)
    assert 'from LSS_python.xismu import xismu' in text
    assert 'migrate_xismu' in text
try:
    tpcf.no_such_attribute
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attributes must raise AttributeError')
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=60)


def test_legacy_function_exports_are_the_same_implementations():
    results = importlib.import_module("LSS_python.xismu")
    from LSS_python import cov, pair_counting, tpcf

    for name in (
        "meannorm", "packarray1d", "packarray2d", "cal_convert_factor",
        "average_xismus_dict_list", "get_diff_array", "get_diff_array_test",
        "get_diff_array_2d",
    ):
        assert getattr(tpcf, name) is getattr(results, name)
    for name in ("compute_DD_JK", "compute_DR_JK", "compute_RR_JK", "extract_npairs"):
        assert getattr(cov, name) is getattr(tpcf, name)
    assert cov.create_xismu_from_pairs is results.create_xismu_from_pairs
    assert cov.get_neighbors is pair_counting.get_neighbors


def test_subsample_counter_keeps_raw_order_and_skips_small_subvolumes(monkeypatch):
    from LSS_python import tpcf

    first = np.ones((3, 4))
    last = np.full((2, 4), 2.0)
    random = np.full((5, 4), 3.0)
    catalogs = [first, None, np.empty((0, 4)), np.ones((1, 4)), last]
    calls = []
    outputs = []

    def count(data1, data2, sedges, mubin, weighted, box, refine, threads, *, autocorr):
        calls.append((data1, data2, autocorr, weighted, box, threads))
        raw = np.array([(7, 0.0)], dtype=[("npairs", "u8"), ("weightavg", "f8")])
        outputs.append(raw)
        return raw

    monkeypatch.setattr(tpcf, "call_DDsmu", count)
    result = tpcf.compute_subsample_pairs(catalogs, random, [0, 1], 1, True, 10)
    assert len(calls) == 5
    assert calls[0][0] is random and calls[0][1] is None and calls[0][2]
    for offset, catalog in ((1, first), (3, last)):
        assert calls[offset][0] is catalog and calls[offset][1] is None
        assert calls[offset][2]
        assert calls[offset + 1][0] is catalog and calls[offset + 1][1] is random
        assert not calls[offset + 1][2]
    assert result["RR_result"] is outputs[0]
    for idx in (1, 2, 3):
        assert result["DD_sub"][idx] is result["DR_sub"][idx] is None
    assert result["DD_sub"][0] is outputs[1]
    assert result["DR_sub"][4] is outputs[4]
    assert all(raw["npairs"][0] == 7 and raw["weightavg"][0] == 0 for raw in outputs)


def test_subsample_counter_allows_separate_rr_cache_phase(monkeypatch):
    from LSS_python import tpcf

    random = np.ones((4, 3))
    raw = np.zeros(1, dtype=[("npairs", "u8")])
    calls = []

    def count(*args, **kwargs):
        calls.append((args, kwargs))
        return raw

    monkeypatch.setattr(tpcf, "call_DDsmu", count)
    rr = tpcf.compute_subsample_pairs(None, random, [0, 1], 1, False, 10, run_parts="RR")
    assert rr == {"DD_sub": {}, "DR_sub": {}, "RR_result": raw}
    assert len(calls) == 1
    dd_dr = tpcf.compute_subsample_pairs(
        [random], random, [0, 1], 1, False, 10, run_parts=("DD", "DR"),
    )
    assert dd_dr["RR_result"] is None
    assert len(calls) == 3


@pytest.mark.parametrize("parts", [("invalid",), (5,)])
def test_subsample_counter_rejects_unknown_parts(parts):
    from LSS_python.tpcf import compute_subsample_pairs

    with pytest.raises(ValueError, match="run_parts"):
        compute_subsample_pairs([], np.ones((3, 3)), [0, 1], 1, False, 10, run_parts=parts)


def test_corrfunc_extraction_preserves_zero_weight_compatibility():
    from LSS_python.tpcf import extract_npairs

    raw = np.array([(2, 0.0), (3, 0.0)], dtype=[("npairs", "u8"), ("weightavg", "f8")])
    np.testing.assert_array_equal(extract_npairs(raw, True), [2, 3])
    raw["weightavg"] = [1.5, 2.0]
    np.testing.assert_array_equal(extract_npairs(raw, True), [3, 6])
    np.testing.assert_array_equal(extract_npairs({"block": raw}, False)["block"], [2, 3])
