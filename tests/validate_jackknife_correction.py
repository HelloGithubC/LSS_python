"""Validate optional Jackknife correction on one staged small weighted catalog.

The output directory must already contain catalogs.npz, baseline_result.pkl
from the pre-change implementation, and its RR.pkl cache. Two focused checks
compare reconstruction with explicit component/region sums and compare serial
with two-rank MPI for None/'mohammad21'. Existing validation output names are
overwritten; staged catalogs, baseline and RR cache remain read-only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import traceback

import joblib
import numpy as np

from validate_resampling_mpi import _compare, _root_check


def independent_samples(baseline, rr_cache, data_list, random_list, sedges, mubin):
    """Sum retained components and ALL region-pair masses without deductions."""
    from LSS_python.xismu import create_xismu_from_pairs

    nregions = len(data_list)
    alpha = nregions / (2. + np.sqrt(2.) * (nregions - 1))
    moments = [(np.array([np.sum(block[:, 3]) for block in blocks]),
                np.array([np.sum(block[:, 3] ** 2) for block in blocks]))
               for blocks in (data_list, random_list)]
    parts = {part: (source[part + '_internal'], source[part + '_cross'])
             for part, source in (('DD', baseline), ('DR', baseline), ('RR', rr_cache))}
    samples = np.empty(nregions, dtype=object)
    for k in range(nregions):
        def keep(i, j):
            if i == k and j == k:
                return 0.
            return 1. - alpha if i == k or j == k else 1.

        histograms, norms = {}, {}
        for part, (internal, cross) in parts.items():
            histogram = np.zeros((len(sedges) - 1, mubin))
            for i, value in internal.items():
                if value is not None:
                    histogram += keep(i, i) * value.reshape(histogram.shape)
            for (i, j), value in cross.items():
                if value is not None:
                    histogram += keep(i, j) * value.reshape(histogram.shape)
            histograms[part] = histogram
            first = moments[1] if part == 'RR' else moments[0]
            second = moments[0] if part == 'DD' else moments[1]
            total = 0.
            # These ordered region pairs deliberately include every region,
            # irrespective of whether it occurs in a spatial cross-count map.
            for i in range(nregions):
                for j in range(nregions):
                    mass = first[0][i] * second[0][j]
                    if part != 'DR' and i == j:
                        mass -= first[1][i]
                    total += keep(i, j) * mass
            norms[part] = total
        samples[k] = create_xismu_from_pairs(
            histograms['DD'], histograms['DR'], histograms['RR'],
            norms['DD'], norms['DR'], norms['RR'], sedges, baseline['muedges'],
        )
    return samples, alpha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--root', type=int, default=1)
    if (argv is None and len(sys.argv) == 1) or argv == []:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    from mpi4py import MPI
    from LSS_python.cov import (
        compute_jackknife_deductions, compute_jackknife_xi,
        get_sub_box_shift, run_jackknife_tpCF,
    )
    from LSS_python.tpcf_mpi import run_jackknife_tpCF_mpi

    comm, root = MPI.COMM_WORLD, args.root
    if not 0 <= root < comm.size:
        parser.error('--root must be a rank in the communicator')
    output = args.output_dir.resolve()
    state, summary = {}, {'ranks': comm.size, 'root': root, 'hosts': comm.allgather(socket.gethostname()),
                          'rtol': 2e-12, 'atol': 1e-10, 'checks': {}}

    def serial_check():
        with np.load(output / 'catalogs.npz') as inputs:
            data, random, edges = (inputs[name] for name in ('data', 'random', 'sedges'))
        baseline = joblib.load(output / 'baseline_result.pkl')
        cache_path = output / 'RR.pkl'
        state['cache_hash'] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        cache = joblib.load(cache_path)
        kwargs = dict(data=data, random=random, sedges=edges, mubin=4, with_weight=True,
                      boxsize=20., ngrids=2, rr_filename=str(cache_path), nthreads=1, full_output=True)
        none = run_jackknife_tpCF(**kwargs)
        default_metrics = {}
        _compare(none, baseline, default_metrics)
        assert all(value == 0. for value in default_metrics.values())
        corrected = run_jackknife_tpCF(**kwargs, correction='mohammad21')
        data_list = list(get_sub_box_shift(data, 20., 2).flat)
        random_list = list(get_sub_box_shift(random, 20., 2).flat)
        expected, alpha = independent_samples(baseline, cache, data_list, random_list, edges, 4)
        oracle_metrics = {}
        _compare(corrected['xismu_jk'], expected, oracle_metrics)
        raw_metrics = {}
        for key in ('xismu_full', 'DD_internal', 'DD_cross', 'DR_internal', 'DR_cross'):
            _compare(corrected[key], none[key], raw_metrics)
        assert all(value == 0. for value in raw_metrics.values())
        maps = [baseline['DD_internal'], baseline['DD_cross'], baseline['DR_internal'],
                baseline['DR_cross'], cache['RR_internal'], cache['RR_cross']]
        # Explicit None also exercises the two lower-level public interfaces.
        rebuilt = compute_jackknife_xi(*maps, data_list, random_list, edges, 4, True,
                                      muedges=baseline['muedges'], correction=None)
        _compare(rebuilt['xismu_jk'], none['xismu_jk'], default_metrics)
        deductions = compute_jackknife_deductions(
            2, *maps, data_list, random_list, edges, 4, True, 2, correction='mohammad21',
        )
        for part, internal, cross in zip(('DD', 'DR', 'RR'), maps[::2], maps[1::2]):
            total = sum((v.reshape(6, 4) for v in (*internal.values(), *cross.values()) if v is not None),
                        start=np.zeros((6, 4)))
            retained = getattr(expected[2], part) * getattr(expected[2], part + 'norm')
            np.testing.assert_allclose(total - deductions[part + '_deduct'], retained, rtol=2e-12, atol=1e-10)
        try:
            compute_jackknife_xi(*maps, data_list, random_list, edges, 4, True, correction='unknown')
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown corrections must raise ValueError')
        for name, result in (('serial_none', none), ('serial_mohammad21', corrected),
                             ('expected_mohammad21', expected)):
            joblib.dump(result, output / (name + '.pkl'))
        summary.update(alpha=alpha, nregions=len(data_list), data_count=len(data), random_count=len(random),
                       boxsize=20., ngrids=2, mubin=4, sedges=edges.tolist(), weighted=True, dtype='float64')
        summary['checks']['reconstruction'] = dict(
            passed=True, default_matches_prechange_exactly=True, default_max_absolute_errors=default_metrics,
            independent_sum_max_absolute_errors=oracle_metrics, raw_components_and_full_sample_unchanged=True,
            unknown_correction_rejected=True,
        )
        state.update(kwargs=kwargs, serial_none=none, serial_mohammad21=corrected)

    _root_check(comm, root, serial_check)
    mpi_metrics = {}
    for correction, label in ((None, 'none'), ('mohammad21', 'mohammad21')):
        kwargs = state['kwargs'] if comm.rank == root else dict(
            data=None, random=None, sedges=None, mubin=None, with_weight=None, boxsize=None, ngrids=None)
        actual = run_jackknife_tpCF_mpi(**kwargs, correction=correction, mpicomm=comm, root=root)
        assert (actual is None) == (comm.rank != root)

        def compare_mpi():
            metrics = {}
            _compare(actual, state['serial_' + label], metrics)
            assert all(value == 0. for value in metrics.values())
            mpi_metrics[label] = metrics
            joblib.dump(actual, output / ('mpi_' + label + '.pkl'))

        _root_check(comm, root, compare_mpi)

    def finish():
        assert hashlib.sha256((output / 'RR.pkl').read_bytes()).hexdigest() == state['cache_hash']
        summary['checks']['mpi'] = dict(passed=True, max_absolute_errors=mpi_metrics,
                                       rr_cache_not_rewritten=True, nonroot_returns_none=True)
        summary['passed'] = True
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)

    _root_check(comm, root, finish)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
