"""Prepare raw catalog marks once for non-AP measurement command lines.

This module does not generate marks or handle pair-count caches. Random
weights are preserved. Gradient vectors are reduced to their Euclidean norm
before data powers are applied once at this boundary.
"""

import numpy as np

from .weight_token import parse_mark_token


def _catalog(source):
    source = np.asarray(source)
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError("catalog must have shape (N, >=3)")
    if source.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("catalog must contain float32 or float64 values")
    if not np.isfinite(source[:, :3]).all():
        raise ValueError("catalog coordinates must be finite")
    return source


def apply_mark_power(mark, power):
    """Evaluate one real power, preserving precision and rejecting invalid weights."""
    if not np.isfinite(mark).all() or not np.isfinite(power):
        raise ValueError("raw marks and power must be finite")
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            result = np.power(mark, power)
    except FloatingPointError as error:
        raise ValueError("raw mark/power produces invalid or overflowing weights") from error
    if not np.isfinite(result).all():
        raise ValueError("powered marks must be finite")
    return result


def _scalar_mark(source, method, offset):
    """Select a scalar mark or reduce gn vectors using the existing Fortran norm."""
    method = parse_mark_token(method)
    columns = source.shape[1] - offset
    if method.kind == "gn" and columns == 3:
        try:
            with np.errstate(over="raise", invalid="raise"):
                # Equivalent to sqrt(gx**2 + gy**2 + gz**2), avoiding square overflow.
                return np.hypot.reduce(source[:, offset:], axis=1)
        except FloatingPointError as error:
            raise ValueError("gradient norm produces invalid or overflowing marks") from error
    if columns != 1:
        expected = "one scalar raw-mark column or three gradient columns" if method.kind == "gn" else "one scalar raw-mark column"
        raise ValueError(f"{method.token} requires {expected}")
    return source[:, offset]


def prepare_box_catalog(source, *, with_weight=False, weight_set_one=False,
                        method="n30", power=1.0, is_data=True, require_weight=False):
    """Convert Cartesian input to three coordinates and one effective weight."""
    source = _catalog(source)
    pos = source[:, :3]
    if not with_weight:
        return np.ascontiguousarray(pos)
    if weight_set_one:
        weight = np.ones(len(source), dtype=source.dtype)
    elif source.shape[1] == 3:
        if require_weight:
            raise ValueError("--with_weight requires a raw-mark/weight column; add it or drop --with_weight")
        weight = np.ones(len(source), dtype=source.dtype)
    elif is_data:
        weight = apply_mark_power(_scalar_mark(source, method, 3), power)
    else:
        weight = source[:, 3]
    result = np.column_stack((pos, weight))
    if not np.isfinite(result).all():
        raise ValueError("enabled weights must be finite")
    return np.ascontiguousarray(result)


def prepare_mock_catalog(source, *, with_weight=False, weight_set_one=False,
                         method="n30", power=1.0, is_data=True, require_mark=False):
    """Preserve survey base weights and combine powered data marks once."""
    source = _catalog(source)
    if source.shape[1] < 4:
        raise ValueError("mock/random catalogs require RA, DEC, R, and a base-weight column")
    if not with_weight:
        return np.ascontiguousarray(source[:, :4])
    if weight_set_one:
        effective = np.ones(len(source), dtype=source.dtype)
    elif not is_data:
        effective = np.prod(source[:, 3:], axis=1)
    elif source.shape[1] == 4:
        if require_mark or power != 1:
            raise ValueError("mock data require a separate raw-mark column after the base weight")
        effective = source[:, 3]
    else:
        if parse_mark_token(method).kind == "gn":
            mark = _scalar_mark(source, method, 4)
            extra = 1.0
        else:
            mark = source[:, 4]
            extra = np.prod(source[:, 5:], axis=1) if source.shape[1] > 5 else 1.0
        effective = source[:, 3] * apply_mark_power(mark, power) * extra
    result = np.column_stack((source[:, :3], effective))
    if not np.isfinite(result).all():
        raise ValueError("enabled mock weights must be finite")
    return np.ascontiguousarray(result)


def attach_weight_metadata(result, args):
    """Record the same declared identity in saved dicts and covariance objects."""
    metadata = {
        "weight_method": args.weight_method if args.with_weight else None,
        "weight_power": args.weight_power_text if args.with_weight else None,
        "weight_token": args.weight_token,
        "weight_set_one": bool(args.with_weight and args.set_weight_to_one),
    }
    if result is None:
        return result
    if isinstance(result, dict):
        result.update(metadata)
    elif isinstance(result, (list, tuple, np.ndarray)):
        for item in result:
            attach_weight_metadata(item, args)
    else:
        for name, value in metadata.items():
            setattr(result, name, value)
    return result
