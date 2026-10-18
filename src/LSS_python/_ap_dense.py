"""Shared preparation and reconstruction for dense AP conversion."""

from dataclasses import dataclass

import numpy as np

from .base import DA, Hz_w0wa


CONVERT_ALGORITHMS = ("polygon", "curved_gauss8", "curved_analytic")


def validate_algorithm(name):
    if name not in CONVERT_ALGORITHMS:
        raise ValueError(f"convert_algorithm must be one of {CONVERT_ALGORITHMS}, got {name!r}")


@dataclass(frozen=True)
class APGeometry:
    redshift: float
    Hz_f: float
    Hz_m: float
    DA_f: float
    DA_m: float

    @property
    def alpha_perp(self):
        return self.DA_m / self.DA_f

    @property
    def alpha_parallel(self):
        return self.Hz_f / self.Hz_m

    @property
    def distance(self):
        return float(np.hypot(self.alpha_perp - 1., self.alpha_parallel - 1.))


def normalize_background_inputs(redshift, omega_mf, w_f, omega_mm, w_m, wa_f=0., wa_m=0.):
    """Use the same finite-input, redshift and constant-w policy for both routes."""
    if wa_f is None or wa_m is None:
        wa_f = wa_m = 0.
    parameters = np.asarray([redshift, omega_mf, w_f, omega_mm, w_m, wa_f, wa_m], dtype=float)
    if not np.all(np.isfinite(parameters)) or redshift < 0:
        raise ValueError("Cosmological parameters must be finite and redshift must be nonnegative")
    return max(float(redshift), 1e-4), wa_f, wa_m


def ap_geometry(redshift, omega_mf, w_f, omega_mm, w_m, wa_f=0., wa_m=0.):
    """Evaluate each distance/rate once, including the constant-w fallback."""
    z, wa_f, wa_m = normalize_background_inputs(redshift, omega_mf, w_f, omega_mm, w_m, wa_f, wa_m)
    geometry = APGeometry(z, float(Hz_w0wa(z, omega_mf, w_f, wa_f)),
                          float(Hz_w0wa(z, omega_mm, w_m, wa_m)),
                          float(DA(z, omega_mf, w_f, wa_f)), float(DA(z, omega_mm, w_m, wa_m)))
    backgrounds = np.array([geometry.Hz_f, geometry.Hz_m, geometry.DA_f, geometry.DA_m])
    if not np.all(np.isfinite(backgrounds)) or np.any(backgrounds <= 0):
        raise RuntimeError("Invalid AP mapping inputs: distances/rates must be finite and positive")
    return geometry


def grid_shape(shape, name):
    if len(shape) != 2 or any(isinstance(v, (bool, np.bool_)) or
                              not isinstance(v, (int, np.integer)) or v <= 0 for v in shape):
        raise ValueError(f"{name} must contain two positive integers")
    if max(shape) > np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds native index limits")
    return tuple(map(int, shape))


def validate_mapping(data, cosmos_tuple, bins, bounds, window, convert_algorithm, nthreads=1):
    """Adapt the historical distance/rate tuple to the ratio-only kernels."""
    validate_algorithm(convert_algorithm)
    data = np.ascontiguousarray(data, dtype=np.float64)
    if data.ndim != 3 or min(data.shape) < 1 or not np.all(np.isfinite(data)):
        raise ValueError("Pair counts must be a finite, nonempty (source_s, source_mu, channels) array")
    if len(bins) != 4:
        raise ValueError("smu_bin_tuple must contain four integers")
    source_shape = grid_shape((bins[0], bins[2]), "source shape")
    target_shape = grid_shape((bins[1], bins[3]), "target shape")
    if data.shape[:2] != source_shape:
        raise ValueError("Input shape does not match the declared source grid")
    background = np.asarray(cosmos_tuple, dtype=float)
    if background.shape != (4,) or not np.all(np.isfinite(background)) or np.any(background <= 0):
        raise RuntimeError("Invalid AP mapping inputs: distances/rates must be finite and positive")
    alpha_perp, alpha_parallel = background[3] / background[2], background[0] / background[1]
    if not np.all(np.isfinite([alpha_perp, alpha_parallel])) or min(alpha_perp, alpha_parallel) <= 0:
        raise RuntimeError("Invalid AP mapping inputs: AP factors must be finite and positive")
    bounds, window = np.asarray(bounds, dtype=float), np.asarray(window, dtype=float)
    if bounds.shape != (4,) or window.shape != (4,) or not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(window)):
        raise ValueError("Grid and mapping bounds must contain four finite values")
    if not 0 <= bounds[0] < bounds[1] or not 0 <= bounds[2] < bounds[3] <= 1:
        raise ValueError("Invalid s/mu grid bounds")
    if window[1] < window[0] or not 0 <= window[2] <= window[3] <= 1:
        raise ValueError("Invalid AP mapping bounds")
    if isinstance(nthreads, (bool, np.bool_)) or not isinstance(nthreads, (int, np.integer)) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer")
    return data, float(alpha_perp), float(alpha_parallel), target_shape, bounds[:2], bounds[2:], window


def validate_xismu(source, assistant=None):
    source_shape = grid_shape((source.sbin, source.mubin), "source shape")
    if not np.isfinite(source.smax) or source.smax <= 0:
        raise ValueError("Source smax must be finite and positive")
    for name in ("DD", "DR", "RR"):
        counts, norm = getattr(source, name, None), getattr(source, name + "norm", None)
        if counts is None or np.shape(counts) != source_shape or not np.all(np.isfinite(counts)):
            raise ValueError(f"Source {name} must be finite with shape {source_shape}")
        if norm is None or not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Source {name}norm must be finite and positive")
    if assistant is None:
        return None
    shape = grid_shape((assistant.sbin, assistant.mubin), "assistant shape")
    if not np.isfinite(assistant.smax) or not np.isclose(source.smax, assistant.smax, rtol=0, atol=1e-12):
        raise ValueError("Source and assistant must share the same smax")
    for name in ("S", "Mu"):
        coordinate = getattr(assistant, name, None)
        if coordinate is None or np.shape(coordinate) != shape or not np.all(np.isfinite(coordinate)):
            raise ValueError(f"Assistant {name} must be finite with shape {shape}")
    return shape


def convert_xismu(source, geometry, sbin2, mubin2, smin_mapping, smax_mapping,
                  assistant, c_api, convert_algorithm):
    """Map normalized pair counts and reconstruct an independent xismu object."""
    validate_algorithm(convert_algorithm)
    assistant_shape = validate_xismu(source, assistant)
    shape = grid_shape((sbin2, mubin2), "target shape")
    if assistant_shape is not None and shape != assistant_shape:
        raise ValueError("Output shape must match the assistant grid")
    if c_api:
        from .CPP.AP import mapping_smudata_dense_cpp as mapper
    else:
        from ._AP_core import mapping_smudata_dense as mapper
    data = np.stack([source.DD, source.DR, source.RR], axis=-1)
    mapped = mapper(data, (geometry.Hz_f, geometry.Hz_m, geometry.DA_f, geometry.DA_m),
                    (source.sbin, sbin2, source.mubin, mubin2), (0., float(source.smax), 0., 1.),
                    (smin_mapping, smax_mapping, 0., 1.), convert_algorithm=convert_algorithm)
    if assistant is None:
        s = (np.arange(sbin2) + .5) * source.smax / sbin2
        mu = (np.arange(mubin2) + .5) / mubin2
        S, Mu = np.meshgrid(s, mu, indexing="ij")
    else:
        S, Mu = np.array(assistant.S, copy=True), np.array(assistant.Mu, copy=True)
    from .tpcf import xismu
    result = xismu(smax=source.smax, sbin=sbin2, mubin=mubin2,
                   DDnorm=source.DDnorm, DRnorm=source.DRnorm, RRnorm=source.RRnorm,
                   DD=mapped[:, :, 0] * source.DDnorm, DR=mapped[:, :, 1] * source.DRnorm,
                   RR=mapped[:, :, 2] * source.RRnorm, S=S, Mu=Mu,
                   **{name: getattr(source, name, None) for name in ("sum_wd", "sum_wr", "sum_wd2", "sum_wr2")})
    result.s_array, result.mu_array = np.nanmean(S, axis=1), np.nanmean(Mu, axis=0)
    return result
