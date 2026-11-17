"""Catalog subsampling and formatting utilities."""

from collections.abc import Mapping
import re

import numpy as np

__all__ = ["format_nbar", "select_by_nbar"]


def format_nbar(nbar: float) -> str:
    """Format a positive number density into a compact tag such as ``'1e-3'``."""
    if isinstance(nbar, bool) or not isinstance(nbar, (int, float, np.number)):
        raise TypeError(f"nbar must be a real number, got {type(nbar).__name__}")
    val = float(nbar)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"nbar must be finite and positive, got {nbar}")

    mantissa, exp = f"{val:.8e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp_sign = exp[0] if exp[0] in ("+", "-") else "+"
    exp_digits = exp[1:].lstrip("0") or "0"
    if exp_sign == "+":
        return f"{mantissa}e{exp_digits}"
    return f"{mantissa}e-{exp_digits}"


def _resolve_volume(boxsize, volume) -> float:
    """Resolve survey/simulation volume with priority ``boxsize > volume``."""
    if boxsize is not None:
        if isinstance(boxsize, bool):
            raise TypeError("boxsize cannot be a bool")
        if isinstance(boxsize, (int, float, np.number)):
            side = float(boxsize)
            if not np.isfinite(side) or side <= 0.0:
                raise ValueError(f"boxsize must be finite and positive, got {boxsize}")
            return side ** 3

        box_arr = np.asarray(boxsize, dtype=np.float64)
        if box_arr.shape != (3,) or not np.all(np.isfinite(box_arr)) or np.any(box_arr <= 0.0):
            raise ValueError(
                "boxsize must be a positive scalar or a 3-element sequence of positive numbers"
            )
        return float(np.prod(box_arr))

    if volume is not None:
        if isinstance(volume, bool) or not isinstance(volume, (int, float, np.number)):
            raise TypeError(f"volume must be a real number, got {type(volume).__name__}")
        vol = float(volume)
        if not np.isfinite(vol) or vol <= 0.0:
            raise ValueError(f"volume must be finite and positive, got {volume}")
        return vol

    raise ValueError("Either boxsize or volume must be provided when n_target is None")


def _resolve_n_target(nbar, boxsize, volume, n_target) -> int:
    """Compute the positive integer target count from ``n_target`` or ``nbar``."""
    if n_target is not None:
        if isinstance(n_target, bool) or not isinstance(n_target, (int, np.integer)):
            raise TypeError(f"n_target must be an integer, got {type(n_target).__name__}")
        target = int(n_target)
        if target < 1:
            raise ValueError(f"n_target must be >= 1, got {n_target}")
        return target

    if nbar is None:
        raise ValueError("Either n_target or nbar must be provided")
    if isinstance(nbar, bool) or not isinstance(nbar, (int, float, np.number)):
        raise TypeError(f"nbar must be a real number, got {type(nbar).__name__}")
    nbar_val = float(nbar)
    if not np.isfinite(nbar_val) or nbar_val <= 0.0:
        raise ValueError(f"nbar must be finite and positive, got {nbar}")

    vol = _resolve_volume(boxsize, volume)
    return max(1, int(np.ceil(nbar_val * vol)))


def _catalog_length(catalog) -> int:
    """Validate supported catalog container and return its row count."""
    if isinstance(catalog, np.ndarray):
        if catalog.ndim == 0:
            raise ValueError("catalog ndarray must be at least 1-D")
        n_total = len(catalog)
    elif isinstance(catalog, Mapping):
        if not catalog:
            raise ValueError("catalog mapping must not be empty")
        lengths = {len(v) for v in catalog.values()}
        if len(lengths) != 1:
            raise ValueError("All arrays in catalog mapping must have the same length")
        n_total = lengths.pop()
    else:
        raise TypeError(
            f"catalog must be a numpy ndarray/memmap or Mapping, got {type(catalog).__name__}"
        )

    if n_total == 0:
        raise ValueError("catalog must not be empty")
    return n_total


def _extract_key_values(catalog, key, n_total: int) -> np.ndarray:
    """Extract a contiguous 1-D key array aligned with ``catalog``."""
    if isinstance(key, bool):
        raise TypeError("key cannot be a bool")

    if isinstance(key, str):
        if isinstance(catalog, np.ndarray) and catalog.dtype.names is None:
            raise TypeError("String key requires a structured ndarray or Mapping catalog")
        values = np.asarray(catalog[key])
    elif isinstance(key, (int, np.integer)):
        if not isinstance(catalog, np.ndarray) or catalog.dtype.names is not None or catalog.ndim != 2:
            raise TypeError("Integer column key is only supported for plain 2-D ndarray catalogs")
        values = np.asarray(catalog[:, int(key)])
    else:
        values = np.asarray(key)

    if values.ndim != 1 or len(values) != n_total:
        raise ValueError(
            f"key array must be 1-D of length {n_total}, got shape {values.shape}"
        )
    if np.issubdtype(values.dtype, np.floating) and np.isnan(values).any():
        raise ValueError("key values contain NaN")
    return values


def _sort_selected_indices(
    indices: np.ndarray,
    values: np.ndarray,
    descending: bool,
) -> np.ndarray:
    """Sort selected indices by ``values`` while preserving relative order on ties."""
    sub_values = values[indices]
    if descending:
        order = (len(sub_values) - 1) - np.argsort(sub_values[::-1], kind="stable")[::-1]
    else:
        order = np.argsort(sub_values, kind="stable")
    return indices[order]


def _select_indices_by_key(
    values: np.ndarray,
    target: int,
    descending: bool,
    sort_output: bool,
) -> np.ndarray:
    """Select top/bottom ``target`` items by ``values``, including boundary ties."""
    n_total = len(values)
    if n_total <= target:
        indices = np.arange(n_total, dtype=np.int64)
        return _sort_selected_indices(indices, values, descending) if sort_output else indices

    if descending:
        kth = n_total - target
        threshold = np.partition(values, kth)[kth]
        indices = np.flatnonzero(values >= threshold)
    else:
        kth = target - 1
        threshold = np.partition(values, kth)[kth]
        indices = np.flatnonzero(values <= threshold)

    if sort_output:
        indices = _sort_selected_indices(indices, values, descending)
    return indices


def _select_indices_random(n_total: int, target: int, seed: int | None) -> np.ndarray:
    """Select exactly ``target`` unique indices uniformly at random."""
    if n_total <= target:
        return np.arange(n_total, dtype=np.int64)

    if seed is None or seed == -1:
        rng = np.random.default_rng()
    elif isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")
    elif int(seed) >= 0:
        rng = np.random.default_rng(int(seed))
    else:
        raise ValueError(f"seed must be >= 0, -1, or None for random sampling, got {seed}")

    return rng.choice(n_total, size=target, replace=False)


def _slice_catalog(catalog, indices: np.ndarray):
    """Materialize selected rows from ``catalog`` preserving container type."""
    if isinstance(catalog, np.ndarray):
        return np.array(catalog[indices], copy=True)
    return {name: np.array(np.asarray(col)[indices], copy=True) for name, col in catalog.items()}


def select_by_nbar(
    catalog,
    *,
    nbar: float | None = None,
    boxsize: float | tuple[float, float, float] | None = None,
    volume: float | None = None,
    n_target: int | None = None,
    key: str | int | np.ndarray | None = None,
    descending: bool = True,
    sort_output: bool = False,
    seed: int | None = None,
    return_indices: bool = False,
):
    """Subsample a catalog to a target number density ``nbar`` or count ``n_target``.

    Parameters
    ----------
    catalog : numpy.ndarray, numpy.memmap, or Mapping
        Input galaxy/halo catalog. File loading should happen outside this function.
    nbar : float, optional
        Target number density in ``(length unit)^-3``. Used with ``boxsize`` or
        ``volume`` when ``n_target`` is not provided.
    boxsize : float or 3-tuple of float, optional
        Cubic box side length or ``(Lx, Ly, Lz)``. Takes priority over ``volume``
        if both are given.
    volume : float, optional
        Effective volume used when ``boxsize`` is not given.
    n_target : int, optional
        Explicit target object count; overrides ``nbar`` / ``boxsize`` / ``volume``.
    key : str, int, or 1-D array-like, optional
        Ranking variable used for selection (e.g. ``"galmass"`` or ``"Mvir"``).
        When provided, all objects at or beyond the boundary threshold (including
        ties) are kept. When ``None``, exactly ``min(n_target, len(catalog))``
        objects are sampled uniformly without replacement.
    descending : bool, default True
        If ``True``, select the largest ``key`` values; otherwise the smallest.
    sort_output : bool, default False
        If ``True`` and ``key`` is provided, sort the selected subset by ``key``.
    seed : int or None, optional
        Random seed used only when ``key is None``. ``>= 0`` fixes the seed;
        ``None`` or ``-1`` draws unseeded random samples.
    return_indices : bool, default False
        If ``True``, return the 1-D integer index array instead of slicing ``catalog``.

    Returns
    -------
    sub_catalog or numpy.ndarray
        Sliced copy of ``catalog`` (or 1-D index array if ``return_indices=True``).
    """
    n_total = _catalog_length(catalog)
    target = _resolve_n_target(nbar=nbar, boxsize=boxsize, volume=volume, n_target=n_target)

    if key is None:
        indices = _select_indices_random(n_total=n_total, target=target, seed=seed)
    else:
        values = _extract_key_values(catalog=catalog, key=key, n_total=n_total)
        indices = _select_indices_by_key(
            values=values,
            target=target,
            descending=descending,
            sort_output=sort_output,
        )

    if return_indices:
        return indices
    return _slice_catalog(catalog, indices)
