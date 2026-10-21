"""Tests for the independent pair-counting core and block orchestration."""

import numpy as np
import pytest
from Corrfunc.theory import DDsmu

from LSS_python.pair_counting import (
    PAIR_TYPES,
    PairCounts,
    PairTask,
    build_pair_tasks,
    count_pairs,
    count_pairs_in_blocks,
    finalize_pair_counts,
    merge_pair_counts,
    pair_normalizations,
    prepare_catalogs,
    split_catalog_into_blocks,
    sum_pair_counts,
)


def whole_catalog_counts(data, random, sedges, mubin, boxsize=None, with_weight=False):
    """Use direct Corrfunc calls and explicit pair weights as the reference."""
    result = {}
    catalogs = (data, random)
    weights = [catalog[:, 3].astype(np.float64)
               if with_weight and catalog.shape[1] == 4 else np.ones(len(catalog))
               for catalog in catalogs]
    for name, i, j in (("DD", 0, 0), ("DR", 0, 1), ("RR", 1, 1)):
        first, second = catalogs[i], catalogs[j]
        kwargs = dict(
            autocorr=i == j, nthreads=1, binfile=sedges, mu_max=1.0,
            nmu_bins=mubin, X1=first[:, 0], Y1=first[:, 1], Z1=first[:, 2],
            periodic=False,
        )
        if boxsize is not None:
            kwargs["boxsize"] = boxsize
        if with_weight:
            kwargs.update(weights1=weights[i].astype(first.dtype), weight_type="pair_product")
        if i != j:
            kwargs.update(X2=second[:, 0], Y2=second[:, 1], Z2=second[:, 2])
            if with_weight:
                kwargs["weights2"] = weights[j].astype(second.dtype)
        raw = DDsmu(**kwargs)
        npairs = raw["npairs"].reshape(len(sedges) - 1, mubin)
        wpairs = npairs.astype(np.float64)
        if with_weight:
            wpairs *= raw["weightavg"].reshape(npairs.shape)
        result[name] = PairCounts(npairs.copy(), wpairs.copy(), np.asarray(sedges).copy(),
                                  np.linspace(0.0, 1.0, mubin + 1), with_weight)
    return result


def assert_counts_match(blocked, whole, rtol=2e-12, atol=1e-10):
    for name in PAIR_TYPES:
        left = blocked[name]
        right = whole[name]
        if isinstance(left, dict):
            np.testing.assert_array_equal(left["npairs"], right.npairs)
            np.testing.assert_allclose(left["wpairs"], right.wpairs, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(left.npairs, right.npairs)
            np.testing.assert_allclose(left.wpairs, right.wpairs, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weight_mode", ["unweighted", "weighted", "missing_column"])
@pytest.mark.parametrize("boxsize,ngrids", [
    (900.0, 3),
    ((900.0, 780.0, 660.0), (2, 3, 2)),
])
def test_random_dd_dr_rr_match_whole_catalog(dtype, weight_mode, boxsize, ngrids):
    rng = np.random.default_rng(20261020)
    box = np.asarray(boxsize)
    if box.ndim == 0:
        box = np.repeat(box, 3)
    data = (rng.random((128, 3)) * box).astype(dtype)
    random = (rng.random((192, 3)) * box).astype(dtype)
    if weight_mode == "weighted":
        data = np.column_stack((data, rng.uniform(0.2, 2.0, len(data)).astype(dtype)))
        random = np.column_stack((random, rng.uniform(0.2, 2.0, len(random)).astype(dtype)))
    with_weight = weight_mode != "unweighted"
    sedges, mubin = np.linspace(0.5, 200.0, 5), 4
    blocked = count_pairs_in_blocks(data, random, sedges, mubin, boxsize, ngrids,
                                    with_weight=with_weight)
    assert all(isinstance(value, PairCounts) for value in blocked["task_counts"].values())
    whole = whole_catalog_counts(data, random, sedges, mubin, boxsize, with_weight)
    tolerance = dict(rtol=3e-6, atol=1e-5) if dtype == np.float32 else {}
    assert_counts_match(blocked, whole, **tolerance)

    whole_norms = pair_normalizations(data, random, with_weight)
    normalized = [
        {name: blocked[name]["wpairs"] / blocked[name]["normalization"]
         for name in PAIR_TYPES},
        {name: whole[name].wpairs / whole_norms[name] for name in PAIR_TYPES},
    ]
    mask = normalized[1]["RR"] > 0
    assert np.any(mask)
    xis = [(source["DD"][mask] - 2 * source["DR"][mask] + source["RR"][mask])
           / source["RR"][mask] for source in normalized]
    np.testing.assert_allclose(xis[0], xis[1], **(tolerance or dict(rtol=2e-12, atol=1e-10)))

    reduced = sum_pair_counts(reversed(list(blocked["task_counts"].items())),
                              sedges, mubin, with_weight)
    for name in PAIR_TYPES:
        np.testing.assert_array_equal(reduced[name].npairs, blocked[name]["npairs"])
        np.testing.assert_allclose(reduced[name].wpairs, blocked[name]["wpairs"],
                                   rtol=2e-12, atol=1e-10)


def test_task_graph_preserves_auto_multiplicity_and_both_dr_directions():
    blocks = tuple(np.empty((1, 3)) for _ in range(27))
    tasks = build_pair_tasks(blocks, blocks, 3)
    task_set = set(tasks)
    assert len(tasks) == len(task_set)
    for task in tasks:
        if task.pair_type == "DR":
            assert not task.autocorr and task.multiplicity == 1
            assert PairTask("DR", task.second, task.first) in task_set
        else:
            assert task.first <= task.second
            assert task.multiplicity == (1 if task.first == task.second else 2)
    for name in ("DD", "RR"):
        assert sum(task.autocorr for task in tasks if task.pair_type == name) == 27
        assert PairTask(name, 0, 18) not in task_set


def test_sparse_blocks_faces_edges_corners_and_box_boundaries():
    data = np.array([
        [0., 0., 0.], [900., 900., 900.],
        [299.9, 299.8, 299.7], [300.1, 300.2, 300.3],
        [299.7, 100.1, 100.2], [300.2, 100.4, 100.6],
        [299.9, 299.8, 100.3], [300.1, 300.4, 100.7],
    ])
    random = np.random.default_rng(9).uniform(0, 900, (11, 3))
    data_before = data.copy()
    edges = np.array([0.01, 1., 100., 200., 300.])
    blocked = count_pairs_in_blocks(data, random, edges, 4, 900., 3)
    whole = whole_catalog_counts(data, random, edges, 4, 900., False)
    assert_counts_match(blocked, whole)
    assert np.count_nonzero(blocked["block_sizes"]["data"] == 0) > 0
    assert blocked["block_sizes"]["data"].sum() == len(data)
    blocks = split_catalog_into_blocks(data, 900., 3)
    np.testing.assert_array_equal(blocks[-1][0], [900., 900., 900.])
    np.testing.assert_array_equal(data, data_before)


def test_zero_separation_keeps_native_counts_without_per_block_fallback():
    data = np.array([[299.9, 40., 60., 2.], [300.2, 40.1, 60.2, 3.]])
    random = np.array([[299.8, 40.3, 60.1, 4.], [300.1, 40.5, 60.4, 5.]])
    edges = np.array([0., 0.1, 1., 100.])
    blocked = count_pairs_in_blocks(data, random, edges, 4, 900., 3, with_weight=True)
    whole = whole_catalog_counts(data, random, edges, 4, 900., True)
    assert_counts_match(blocked, whole)
    assert blocked["DD"]["npairs"][0].sum() == 2
    assert blocked["DD"]["wpairs"][0].sum() == 0


def test_empty_catalogs_produce_zero_histograms_without_tasks():
    empty = np.empty((0, 3))
    result = count_pairs_in_blocks(empty, empty, [0.1, 100.], 4, 900., 3,
                                   with_weight=True)
    assert not result["tasks"]
    for name in PAIR_TYPES:
        assert not np.any(result[name]["npairs"])
        assert not np.any(result[name]["wpairs"])
        assert result[name]["normalization"] == 0


def test_zero_weightavg_compatibility_runs_only_after_complete_reduction():
    edges = np.array([0., 1.])
    medges = np.array([0., 1.])
    task = PairTask("DD", 0, 0)
    raw = PairCounts(np.array([[3]], dtype=np.uint64), np.zeros((1, 1)),
                     edges, medges, True)
    reduced = sum_pair_counts([(task, raw)], edges, 1, with_weight=True)
    np.testing.assert_array_equal(reduced["DD"].wpairs, [[0.]])
    finalized = finalize_pair_counts(reduced)
    np.testing.assert_array_equal(finalized["DD"]["wpairs"], [[3.]])
    with pytest.raises(ValueError, match="duplicate"):
        sum_pair_counts([(task, raw), (task, raw)], edges, 1, with_weight=True)


def test_neighbor_stencil_rejects_separations_larger_than_block_width():
    catalog = np.random.default_rng(2).uniform(0, 900., (12, 3))
    with pytest.raises(ValueError, match="smallest block width"):
        count_pairs_in_blocks(catalog, catalog, [0.1, 301.], 4, 900., 3)


def test_mixed_catalog_precision_uses_common_corrfunc_dtype():
    rng = np.random.default_rng(10)
    data = rng.uniform(0, 900., (32, 3)).astype(np.float32)
    random = rng.uniform(0, 900., (48, 3))
    edges = np.array([0.1, 100., 200.])
    result = count_pairs_in_blocks(data, random, edges, 4, 900., 3)
    whole = whole_catalog_counts(data.astype(np.float64), random, edges, 4, 900., False)
    assert_counts_match(result, whole)


def test_core_matches_corrfunc_for_irregular_negative_catalogs_and_random_partitions():
    rng = np.random.default_rng(812)
    data = np.column_stack((rng.normal(-2.0, 3.0, (31, 3)), rng.uniform(0.2, 2.5, 31)))
    random = np.column_stack((rng.normal(1.0, 4.0, (37, 3)), rng.uniform(0.3, 1.8, 37)))
    edges, mubin = np.array([0.0, 1.0, 3.5, 8.0]), 5
    direct = whole_catalog_counts(data, random, edges, mubin, with_weight=True)
    for actual, expected in (("DD", direct["DD"]), ("DR", direct["DR"]), ("RR", direct["RR"])):
        first, second, autocorr = {
            "DD": (data, None, True),
            "DR": (data, random, False),
            "RR": (random, None, True),
        }[actual]
        observed = count_pairs(first, second, autocorr=autocorr, sedges=edges,
                               mubin=mubin, with_weight=True)
        np.testing.assert_array_equal(observed.npairs, expected.npairs)
        np.testing.assert_allclose(observed.wpairs, expected.wpairs, rtol=1e-12, atol=1e-12)

    # Independently partition each catalog at arbitrary indices; combine all ordered
    # cross-partitions and unordered auto-partitions using explicit multiplicities.
    dparts = [data[idx] for idx in np.array_split(rng.permutation(len(data)), [3, 12, 29])]
    rparts = [random[idx] for idx in np.array_split(rng.permutation(len(random)), [8, 13, 30])]
    pieces = {name: [] for name in PAIR_TYPES}
    for i, left in enumerate(dparts):
        for j, right in enumerate(dparts[i:], start=i):
            pieces["DD"].append((count_pairs(left, right if i != j else None,
                                                autocorr=i == j, sedges=edges,
                                                mubin=mubin, with_weight=True),
                                 1 if i == j else 2))
    for left in dparts:
        for right in rparts:
            pieces["DR"].append((count_pairs(left, right, autocorr=False, sedges=edges,
                                                mubin=mubin, with_weight=True), 1))
    for i, left in enumerate(rparts):
        for j, right in enumerate(rparts[i:], start=i):
            pieces["RR"].append((count_pairs(left, right if i != j else None,
                                                autocorr=i == j, sedges=edges,
                                                mubin=mubin, with_weight=True),
                                 1 if i == j else 2))
    merged = merge_pair_counts(pieces, sedges=edges, mubin=mubin, with_weight=True)
    assert_counts_match(merged, direct, rtol=1e-12, atol=1e-12)


def test_merge_explicit_multiplicity_order_two_level_and_input_immutability():
    sedges, mubin = np.array([0., 1., 2.]), 2
    medges = np.linspace(0., 1., mubin + 1)
    def contribution(n, w):
        return PairCounts(np.full((2, 2), n, dtype=np.uint64),
                          np.full((2, 2), w, dtype=np.float64),
                          sedges.copy(), medges.copy(), True)

    a, b, c = contribution(1, 0.25), contribution(3, 0.75), contribution(2, 1.5)
    snapshots = [(x.npairs.copy(), x.wpairs.copy(), x.sedges.copy(), x.muedges.copy())
                 for x in (a, b, c)]
    left = merge_pair_counts({name: [(a, 1), (b, 2)] for name in PAIR_TYPES},
                             sedges=sedges, mubin=mubin, with_weight=True)
    right = merge_pair_counts({name: [(c, 3)] for name in PAIR_TYPES},
                              sedges=sedges, mubin=mubin, with_weight=True)
    result = merge_pair_counts({name: [(right[name], 1), (left[name], 1)]
                                for name in PAIR_TYPES},
                               sedges=sedges, mubin=mubin, with_weight=True)
    for name in PAIR_TYPES:
        np.testing.assert_array_equal(left[name].npairs, np.full((2, 2), 7))
        np.testing.assert_allclose(left[name].wpairs, np.full((2, 2), 1.75))
        np.testing.assert_array_equal(result[name].npairs, np.full((2, 2), 13))
        np.testing.assert_allclose(result[name].wpairs, np.full((2, 2), 6.25))
        assert result[name] is not left[name]
        assert result[name].npairs is not left[name].npairs
    for x, snapshot in zip((a, b, c), snapshots):
        for value, expected in zip((x.npairs, x.wpairs, x.sedges, x.muedges), snapshot):
            np.testing.assert_array_equal(value, expected)


def test_merge_rejects_incompatible_bins_shapes_weights_and_invalid_multiplicity():
    sedges, medges = np.array([0., 1., 2.]), np.array([0., 0.5, 1.])
    good = PairCounts(np.ones((2, 2), dtype=np.uint64), np.ones((2, 2)),
                      sedges, medges, False)
    bad_s = PairCounts(good.npairs.copy(), good.wpairs.copy(), np.array([0., 0.7, 2.]),
                       medges, False)
    bad_mu = PairCounts(good.npairs.copy(), good.wpairs.copy(), sedges,
                        np.array([0., 0.4, 1.]), False)
    bad_weight = PairCounts(good.npairs.copy(), good.wpairs.copy(), sedges, medges, True)
    for bad in (bad_s, bad_mu, bad_weight):
        with pytest.raises(ValueError):
            merge_pair_counts({name: [(bad, 1)] for name in PAIR_TYPES},
                              sedges=sedges, mubin=2, with_weight=False)
    with pytest.raises(ValueError):
        merge_pair_counts({name: [(good, -1)] for name in PAIR_TYPES},
                          sedges=sedges, mubin=2)
    with pytest.raises(ValueError):
        merge_pair_counts({name: [(good, 1.5)] for name in PAIR_TYPES},
                          sedges=sedges, mubin=2)


def test_empty_merge_and_empty_catalog_finalization():
    sedges = np.array([0.1, 1., 2.])
    merged = merge_pair_counts({name: [] for name in PAIR_TYPES},
                               sedges=sedges, mubin=3, with_weight=True)
    for counts in merged.values():
        np.testing.assert_array_equal(counts.npairs, np.zeros((2, 3), dtype=np.uint64))
        np.testing.assert_array_equal(counts.wpairs, np.zeros((2, 3)))
    empty = np.empty((0, 3))
    high_level = count_pairs_in_blocks(empty, empty, sedges, 3, 900., 3, with_weight=True)
    assert high_level["tasks"] == ()
    for name in PAIR_TYPES:
        np.testing.assert_array_equal(high_level[name]["wpairs"], np.zeros((2, 3)))
        assert high_level[name]["normalization"] == 0


def test_prepare_catalogs_unifies_precision_and_adds_unit_weights_without_mutation():
    data = np.arange(15, dtype=np.float32).reshape(5, 3)
    random = np.arange(21, dtype=np.float64).reshape(7, 3)
    data_before, random_before = data.copy(), random.copy()
    prepared_data, prepared_random = prepare_catalogs(data, random, with_weight=True)
    assert prepared_data.dtype == prepared_random.dtype == np.float64
    np.testing.assert_array_equal(prepared_data[:, 3], np.ones(len(data)))
    np.testing.assert_array_equal(prepared_random[:, 3], np.ones(len(random)))
    np.testing.assert_array_equal(data, data_before)
    np.testing.assert_array_equal(random, random_before)


def test_finalize_and_normalization_return_expected_values():
    data = np.array([[0., 0., 0., 2.], [1., 0., 0., 3.]])
    random = np.array([[0., 1., 0., 4.]])
    norms = pair_normalizations(data, random, with_weight=True)
    assert norms == {"DD": 12., "DR": 20., "RR": 0.}
    counts = merge_pair_counts({name: [] for name in PAIR_TYPES},
                               sedges=[0., 1.], mubin=1, with_weight=True)
    finalized = finalize_pair_counts(counts)
    for name in PAIR_TYPES:
        assert set(finalized[name]) == {"npairs", "wpairs"}
        assert finalized[name]["npairs"].shape == (1, 1)
        assert finalized[name]["wpairs"].shape == (1, 1)
