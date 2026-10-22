"""MPI orchestration for the non-periodic ``run_tpCF`` pair-counting interface.

All ranks call run_tpCF_mpi collectively. The designated root prepares spatial
blocks and tasks; ranks receive the blocks needed by their assigned tasks and
use pair_counting's serial cores. Catalog arrays travel in bounded MPI buffer
messages; only small metadata uses pickle. Raw histograms are reduced
with MPI SUM, then root returns Corrfunc-compatible structured arrays and writes
requested files. MPI is imported lazily, keeping it an optional dependency.
"""

from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np

from .pair_counting import (
    PAIR_TYPES,
    PairCounts,
    build_pair_tasks,
    count_pair_task,
    prepare_catalogs,
    split_catalog_into_blocks,
    sum_pair_counts,
)


__all__ = ["run_tpCF_mpi"]

_TRANSFER_CHUNK_BYTES = 256 * 1024**2


def _array_layout(array):
    """Describe a contiguous numeric buffer without including its contents."""
    if not array.flags.c_contiguous or array.dtype.hasobject:
        raise ValueError("MPI array transfers require C-contiguous, non-object arrays")
    dtype = array.dtype.descr if array.dtype.names else array.dtype.str
    return array.shape, dtype


def _buffer_chunks(array):
    """Yield byte views with bounded MPI counts, without copying the array."""
    buffer = array.reshape(-1).view(np.uint8)
    for start in range(0, buffer.size, _TRANSFER_CHUNK_BYTES):
        yield buffer[start:start + _TRANSFER_CHUNK_BYTES]


def _check_receive_allocation(comm, error, stage):
    """Make allocation failures collective before sending any array buffers."""
    errors = comm.allgather(error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors)
                if message is not None]
    if failures:
        raise RuntimeError(f"MPI {stage} failed: " + "; ".join(failures))


def _abort_transfer(comm, exc):
    """Terminate peers when an in-progress buffer transfer cannot complete."""
    try:
        print(f"[MPI rank {comm.Get_rank()}] Buffer transfer failed: {_failure(exc)}",
              file=sys.stderr, flush=True)
    finally:
        comm.Abort(1)


def _distribute_work(comm, packages, root):
    """Scatter small task/layout metadata, then send catalog buffers in chunks."""
    from mpi4py import MPI

    rank = comm.Get_rank()
    metadata, error = None, None
    if rank == root:
        try:
            metadata = [
                (tasks, *({key: _array_layout(block) for key, block in blocks.items()}
                          for blocks in (data, random)))
                for tasks, data, random in packages
            ]
        except Exception as exc:
            error = _failure(exc)
    error = comm.bcast(error, root=root)
    if error is not None:
        raise RuntimeError(f"MPI catalog metadata failed on rank {root}: {error}")
    tasks, data_layouts, random_layouts = comm.scatter(metadata, root=root)

    local, error = None, None
    try:
        if rank == root:
            local = packages[root]
        else:
            data, random = (
                {key: np.empty(shape, dtype=np.dtype(dtype))
                 for key, (shape, dtype) in layouts.items()}
                for layouts in (data_layouts, random_layouts)
            )
            local = tasks, data, random
    except Exception as exc:
        error = _failure(exc)
    _check_receive_allocation(comm, error, "catalog receive allocation")

    try:
        if rank == root:
            for destination, (_, data, random) in enumerate(packages):
                if destination == root:
                    continue
                for blocks in (data, random):
                    for block in blocks.values():
                        for chunk in _buffer_chunks(block):
                            comm.Send([chunk, MPI.BYTE], dest=destination, tag=0)
        else:
            for blocks in local[1:]:
                for block in blocks.values():
                    for chunk in _buffer_chunks(block):
                        comm.Recv([chunk, MPI.BYTE], source=root, tag=0)
    except BaseException as exc:
        _abort_transfer(comm, exc)
        raise
    return local


def _broadcast_result(comm, result, root):
    """Broadcast result layouts and bounded buffers, including structured dtypes."""
    from mpi4py import MPI

    rank = comm.Get_rank()
    metadata, error = None, None
    if rank == root:
        try:
            metadata = {name: None if result[name] is None else _array_layout(result[name])
                        for name in PAIR_TYPES}
        except Exception as exc:
            error = _failure(exc)
    metadata, error = comm.bcast((metadata, error), root=root)
    if error is not None:
        raise RuntimeError(f"MPI result metadata failed on rank {root}: {error}")

    error = None
    if rank != root:
        try:
            result = {name: None if layout is None else
                      np.empty(layout[0], dtype=np.dtype(layout[1]))
                      for name, layout in metadata.items()}
        except Exception as exc:
            error = _failure(exc)
    _check_receive_allocation(comm, error, "result receive allocation")
    try:
        for name in PAIR_TYPES:
            if result[name] is not None:
                for chunk in _buffer_chunks(result[name]):
                    comm.Bcast([chunk, MPI.BYTE], root=root)
    except BaseException as exc:
        _abort_transfer(comm, exc)
        raise
    return result


def _log(message, verbose, rank):
    if verbose:
        print(f"[MPI rank {rank}] {message}", flush=True)


def _positive_integer(value, name):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or value < 1):
        raise ValueError(f"{name} must be a positive integer")


def _run_parts(run_parts):
    parts = [run_parts] if isinstance(run_parts, str) else list(run_parts)
    if any(not isinstance(part, str) for part in parts):
        raise ValueError("run_parts must contain DD, DR, RR, or all")
    parts = {part.upper() for part in parts}
    if parts - set(PAIR_TYPES) - {"ALL"}:
        raise ValueError("run_parts must contain DD, DR, RR, or all")
    return tuple(name for name in PAIR_TYPES if "ALL" in parts or name in parts)


def _assign_tasks(tasks, data_blocks, random_blocks, size):
    """Balance first-block task groups while reusing their input blocks.

    Costs approximate candidate pair products, rather than measured Corrfunc
    runtime. Each task is assigned exactly once; root participates like any rank.
    """
    groups = {}
    for task in tasks:
        first_blocks = random_blocks if task.pair_type == "RR" else data_blocks
        second_blocks = data_blocks if task.pair_type == "DD" else random_blocks
        nfirst, nsecond = len(first_blocks[task.first]), len(second_blocks[task.second])
        cost = nfirst * (nfirst - 1) // 2 if task.autocorr else nfirst * nsecond
        group = groups.setdefault(task.first, [0, []])
        group[0] += max(1, cost)
        group[1].append(task)

    assigned = [[] for _ in range(size)]
    loads = [0] * size
    for first, (cost, group_tasks) in sorted(groups.items(), key=lambda item: (-item[1][0], item[0])):
        rank = min(range(size), key=lambda r: (loads[r], len(assigned[r]), r))
        assigned[rank].extend(group_tasks)
        loads[rank] += cost

    packages = []
    for rank_tasks in assigned:
        data_ids, random_ids = set(), set()
        for task in rank_tasks:
            if task.pair_type == "DD":
                data_ids.update((task.first, task.second))
            elif task.pair_type == "DR":
                data_ids.add(task.first)
                random_ids.add(task.second)
            else:
                random_ids.update((task.first, task.second))
        packages.append((
            tuple(rank_tasks),
            {i: data_blocks[i] for i in sorted(data_ids)},
            {i: random_blocks[i] for i in sorted(random_ids)},
        ))
    return packages


def _count_local_tasks(tasks, data_blocks, random_blocks, config, rank):
    """Yield unmodified task results with progress at task boundaries."""
    total = len(tasks)
    started = last_report = perf_counter()
    report_every = max(1, (total + 19) // 20)
    verbose = config["verbose"]
    _log(f"Counting {total:,} assigned tasks ...", verbose, rank)
    for completed, task in enumerate(tasks, 1):
        task_started = perf_counter()
        report = completed == 1 or (completed - 1) % report_every == 0
        if verbose and report:
            first_blocks = random_blocks if task.pair_type == "RR" else data_blocks
            second_blocks = data_blocks if task.pair_type == "DD" else random_blocks
            mode = "auto" if task.autocorr else "cross"
            _log(f"Starting task {completed:,}/{total:,}: {task.pair_type} {mode}, "
                 f"blocks=({task.first}, {task.second}), "
                 f"points=({len(first_blocks[task.first]):,}, {len(second_blocks[task.second]):,})",
                 verbose, rank)
        result = count_pair_task(
            task, data_blocks, random_blocks, config["sedges"], config["mubin"],
            config["with_weight"], config["boxsize"], nthreads=config["nthreads"],
            refine_factors=config["refine_factors"],
        )
        now = perf_counter()
        if verbose and (report or completed == total or now - last_report >= 30.0):
            _log(f"Task progress {completed:,}/{total:,} ({100.0 * completed / total:.1f}%): "
                 f"{task.pair_type} blocks=({task.first}, {task.second}), "
                 f"task={now - task_started:.2f}s, elapsed={now - started:.2f}s",
                 verbose, rank)
            last_report = now
        yield result
    _log(f"All {total:,} local tasks counted; merging raw histograms ...", verbose, rank)


def _prepare_work(data, random, sedges, mubin, with_weight, run_parts,
                  refine_factors, output_dict, nthreads, verbose, boxsize,
                  ngrids, size, broadcast_result, root=0):
    """Validate root inputs and prepare metadata plus per-rank block packages."""
    parts = _run_parts(run_parts)
    _positive_integer(mubin, "mubin")
    _positive_integer(nthreads, "nthreads")
    if not isinstance(with_weight, (bool, np.bool_)):
        raise ValueError("with_weight must be boolean")
    if len(refine_factors) != 3:
        raise ValueError("refine_factors must contain three positive integers")
    for factor in refine_factors:
        _positive_integer(factor, "refine_factors")
    edges = np.asarray(sedges, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges))
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("sedges must be finite, non-negative, and strictly increasing")
    box = np.asarray(boxsize, dtype=np.float64)
    if box.ndim == 0:
        box = np.repeat(box, 3)
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0):
        raise ValueError("boxsize must contain one or three finite positive lengths")
    grid = np.asarray(ngrids)
    if grid.ndim == 0:
        grid = np.repeat(grid, 3)
    if (grid.shape != (3,) or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid <= 0)):
        raise ValueError("ngrids must contain one or three positive integers")
    if edges[-1] > np.min(box / grid):
        raise ValueError("smax must not exceed the smallest block width")

    need_data = "DD" in parts or "DR" in parts
    need_random = "RR" in parts or "DR" in parts
    if need_data and data is None:
        raise ValueError("requested DD/DR requires data_catalog on root")
    if need_random and random is None:
        raise ValueError("requested DR/RR requires random_catalog on root")
    _log("Validating catalog inputs and preparing columns/precision ...", verbose, root)
    if need_data and need_random:
        data, random = prepare_catalogs(data, random, with_weight=with_weight)
    elif need_data:
        data, _ = prepare_catalogs(data, with_weight=with_weight)
    elif need_random:
        random, _ = prepare_catalogs(random, with_weight=with_weight)
    empty = np.empty((0, 4 if with_weight else 3), dtype=np.float64)
    nblocks = int(np.prod(grid))
    _log(f"Partition: grid={grid.tolist()}, blocks={nblocks:,}, "
         f"block widths={(box / grid).tolist()}, smax={edges[-1]:g}", verbose, root)
    catalogs = []
    for label, catalog, needed in (("data", data, need_data), ("random", random, need_random)):
        if not needed:
            catalogs.append((empty,) * nblocks)
            continue
        _log(f"Splitting {label}: {len(catalog):,} points, dtype={catalog.dtype} ...", verbose, root)
        started = perf_counter()
        blocks = split_catalog_into_blocks(catalog, box, grid, with_weight)
        catalogs.append(blocks)
        if verbose:
            populations = np.array([len(block) for block in blocks])
            _log(f"{label} split complete in {perf_counter() - started:.2f}s: "
                 f"nonempty blocks={np.count_nonzero(populations):,}/{nblocks:,}, "
                 f"points per block min/mean/max="
                 f"{populations.min():,}/{populations.mean():.1f}/{populations.max():,}", verbose, root)
    data_blocks, random_blocks = catalogs
    _log("Building internal and neighboring block-pair tasks ...", verbose, root)
    tasks = tuple(task for task in build_pair_tasks(data_blocks, random_blocks, grid)
                  if task.pair_type in parts)
    if verbose:
        for name in parts:
            selected = [task for task in tasks if task.pair_type == name]
            internal = sum(task.first == task.second for task in selected)
            _log(f"{name} tasks: internal={internal:,}, cross-block={len(selected) - internal:,}, "
                 f"total={len(selected):,}", verbose, root)
    _log(f"Assigning {len(tasks):,} tasks to {size} ranks ...", verbose, root)
    packages = _assign_tasks(tasks, data_blocks, random_blocks, size)
    if verbose:
        for rank, (rank_tasks, rank_data, rank_random) in enumerate(packages):
            breakdown = ", ".join(f"{name}={sum(task.pair_type == name for task in rank_tasks):,}"
                                  for name in parts)
            payload = sum(block.nbytes for blocks in (rank_data, rank_random) for block in blocks.values())
            _log(f"Assignment for rank {rank}: tasks={len(rank_tasks):,} ({breakdown}), "
                 f"data/random blocks={len(rank_data):,}/{len(rank_random):,}, "
                 f"catalog payload={payload / 1024**3:.3f} GiB", verbose, root)
    outputs = {name: None if output_dict is None else output_dict.get(name) for name in parts}
    config = {
        "sedges": edges, "mubin": int(mubin), "with_weight": bool(with_weight),
        "parts": parts, "boxsize": box, "refine_factors": tuple(refine_factors),
        "nthreads": int(nthreads), "verbose": bool(verbose), "outputs": outputs,
        "broadcast_result": bool(broadcast_result),
    }
    return config, packages


def _corrfunc_result(counts):
    """Restore the default DDsmu structured output from a global raw histogram."""
    result = np.zeros(counts.npairs.size, dtype=[
        ("smin", np.float64), ("smax", np.float64), ("savg", np.float64),
        ("mu_max", np.float64), ("npairs", np.uint64), ("weightavg", np.float64),
    ])
    mubin = counts.npairs.shape[1]
    result["smin"] = np.repeat(counts.sedges[:-1], mubin)
    result["smax"] = np.repeat(counts.sedges[1:], mubin)
    result["mu_max"] = np.tile(np.arange(1, mubin + 1) * (1.0 / mubin), len(counts.sedges) - 1)
    result["npairs"] = counts.npairs.ravel()
    if counts.with_weight:
        np.divide(counts.wpairs.ravel(), result["npairs"], out=result["weightavg"],
                  where=result["npairs"] != 0)
    return result


def _failure(exc):
    return f"{type(exc).__name__}: {exc}"


def run_tpCF_mpi(data_catalog, random_catalog, sedges, mubin, with_weight,
                 run_parts=("all",), refine_factors=(2, 2, 1), output_dict=None,
                 nthreads=1, verbose=False, *, boxsize, ngrids,
                 mpicomm=None, root=0, broadcast_result=False):
    """Collectively compute run_tpCF-compatible DD/DR/RR using MPI block tasks.

    All ranks must call this function with the same communicator and root. Other
    configuration and catalogs are taken only from root; non-root catalogs may
    be None. boxsize and ngrids describe the non-periodic [0, boxsize] partition,
    with smax <= min(boxsize / ngrids). nthreads is the OpenMP count per rank.

    Requested entries are default Corrfunc structured arrays; unused entries are
    None. Only root returns the dictionary, unless broadcast_result=True. Other
    ranks otherwise return None. Only root writes output_dict files. Root input,
    Python worker, and output exceptions are propagated to every rank before the
    next collective phase. Fatal native/MPI failures remain MPI runtime errors.

    Raw pair weights are reduced before constructing weightavg. There is no
    normalization, local all-zero fallback, or special statistical correction;
    existing consumers retain their complete-result weight compatibility rule.
    Verbose logs describe preparation, distribution, per-rank task progress,
    reduction, and output. Progress is reported at completed task boundaries.
    Catalogs and optional result broadcasts use direct buffer messages of at
    most 256 MiB; task/layout metadata is sent separately using pickle.
    """
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("run_tpCF_mpi requires the optional mpi4py dependency") from exc
    comm = MPI.COMM_WORLD if mpicomm is None else mpicomm
    rank, size = comm.Get_rank(), comm.Get_size()
    if (not isinstance(root, (int, np.integer)) or isinstance(root, (bool, np.bool_))
            or not 0 <= root < size):
        raise ValueError("root must be a valid rank in mpicomm")

    config, packages, error = None, None, None
    if rank == root:
        try:
            prepare_started = perf_counter()
            _log("Preparing MPI block counting ...", verbose, root)
            config, packages = _prepare_work(
                data_catalog, random_catalog, sedges, mubin, with_weight, run_parts,
                refine_factors, output_dict, nthreads, verbose, boxsize, ngrids,
                size, broadcast_result, root=root,
            )
            if config["verbose"]:
                _log(f"Preparation finished in {perf_counter() - prepare_started:.2f}s; "
                     f"tasks per rank={[len(package[0]) for package in packages]}; "
                     f"OpenMP threads per rank={config['nthreads']}", True, root)
        except Exception as exc:
            error = _failure(exc)
    config, error = comm.bcast((config, error), root=root)
    if error is not None:
        raise RuntimeError(f"MPI preparation failed on rank {root}: {error}")

    if rank == root:
        _log("Distributing task metadata and catalog buffers "
             f"(messages <= {_TRANSFER_CHUNK_BYTES / 1024**2:g} MiB) ...",
             config["verbose"], root)
    distributed_at = perf_counter()
    tasks, data_blocks, random_blocks = _distribute_work(comm, packages, root)
    del packages
    _log(f"Received {len(tasks):,} tasks on {MPI.Get_processor_name()}, "
         f"data/random blocks={len(data_blocks):,}/{len(random_blocks):,}; "
         f"distribution elapsed={perf_counter() - distributed_at:.2f}s", config["verbose"], rank)
    local, error = None, None
    try:
        local_started = perf_counter()
        local = sum_pair_counts(
            _count_local_tasks(tasks, data_blocks, random_blocks, config, rank),
            config["sedges"], config["mubin"], config["with_weight"],
        )
        _log(f"Local counting and merge finished in {perf_counter() - local_started:.2f}s; "
             "waiting for other ranks ...", config["verbose"], rank)
    except Exception as exc:
        error = _failure(exc)
    errors = comm.allgather(error)
    failures = [f"rank {i}: {message}" for i, message in enumerate(errors) if message is not None]
    if failures:
        raise RuntimeError("MPI pair counting failed: " + "; ".join(failures))

    local_npairs = np.stack([local[name].npairs for name in PAIR_TYPES])
    local_wpairs = np.stack([local[name].wpairs for name in PAIR_TYPES])
    global_npairs = np.empty_like(local_npairs) if rank == root else None
    global_wpairs = np.empty_like(local_wpairs) if rank == root else None
    reduce_started = perf_counter()
    if rank == root:
        _log("All ranks finished; reducing raw integer counts and weighted sums ...", config["verbose"], root)
    comm.Reduce(local_npairs, global_npairs, op=MPI.SUM, root=root)
    comm.Reduce(local_wpairs, global_wpairs, op=MPI.SUM, root=root)

    result, error = None, None
    if rank == root:
        try:
            _log(f"MPI reduction finished in {perf_counter() - reduce_started:.2f}s; "
                 "restoring Corrfunc output arrays ...", config["verbose"], root)
            result = dict.fromkeys(PAIR_TYPES)
            muedges = np.linspace(0.0, 1.0, config["mubin"] + 1)
            for i, name in enumerate(PAIR_TYPES):
                if name not in config["parts"]:
                    continue
                counts = PairCounts(global_npairs[i], global_wpairs[i], config["sedges"],
                                    muedges, config["with_weight"])
                result[name] = _corrfunc_result(counts)
                filename = config["outputs"][name]
                if filename is not None:
                    save_started = perf_counter()
                    _log(f"Saving {name} to {filename} ...", config["verbose"], root)
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(result[name], filename)
                    _log(f"Saved {name} in {perf_counter() - save_started:.2f}s", config["verbose"], root)
            _log("MPI pair counting complete.", config["verbose"], root)
        except Exception as exc:
            error = _failure(exc)
    error = comm.bcast(error, root=root)
    if error is not None:
        raise RuntimeError(f"MPI output failed on rank {root}: {error}")
    if config["broadcast_result"]:
        return _broadcast_result(comm, result, root)
    return result
