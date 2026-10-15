"""Numba implementation of conservative polygon AP remapping."""

import numpy as np
from numba import njit


@njit
def _area(points, count):
    if count < 3:
        return 0.0
    # Triangulate relative to the first vertex to reduce cancellation.
    s0 = points[0, 0]
    m0 = points[0, 1]
    twice = 0.0
    for i in range(1, count - 1):
        twice += ((points[i, 0] - s0) * (points[i + 1, 1] - m0)
                  - (points[i + 1, 0] - s0) * (points[i, 1] - m0))
    return 0.5 * abs(twice)


@njit
def polygon_rate(corners, bounds):
    """Return the clipped area fraction for BL, BR, TL, TR quad corners."""
    # The clipping walk uses perimeter order BL, BR, TR, TL.
    first = np.empty((12, 2), dtype=np.float64)
    second = np.empty((12, 2), dtype=np.float64)
    order = (0, 1, 3, 2)
    smin = corners[0, 0]
    smax = smin
    mmin = corners[0, 1]
    mmax = mmin
    for i in range(4):
        source = order[i]
        first[i, 0] = corners[source, 0]
        first[i, 1] = corners[source, 1]
        smin = min(smin, corners[i, 0])
        smax = max(smax, corners[i, 0])
        mmin = min(mmin, corners[i, 1])
        mmax = max(mmax, corners[i, 1])
    if smax <= bounds[0] or smin >= bounds[1] or mmax <= bounds[2] or mmin >= bounds[3]:
        return 0.0
    denominator = _area(first, 4)
    if not denominator > 0.0:
        return 0.0
    if smin >= bounds[0] and smax <= bounds[1] and mmin >= bounds[2] and mmax <= bounds[3]:
        return 1.0

    active = 0
    count = 4
    for edge in range(4):
        if count == 0:
            break
        if active == 0:
            inp, out = first, second
        else:
            inp, out = second, first
        axis = edge // 2
        above = edge % 2 == 0
        bound = bounds[edge]
        previous_s = inp[count - 1, 0]
        previous_m = inp[count - 1, 1]
        previous_value = previous_s if axis == 0 else previous_m
        previous_inside = previous_value >= bound if above else previous_value <= bound
        next_count = 0
        for i in range(count):
            current_s = inp[i, 0]
            current_m = inp[i, 1]
            current_value = current_s if axis == 0 else current_m
            current_inside = current_value >= bound if above else current_value <= bound
            if previous_inside != current_inside:
                t = (bound - previous_value) / (current_value - previous_value)
                out[next_count, 0] = previous_s + t * (current_s - previous_s)
                out[next_count, 1] = previous_m + t * (current_m - previous_m)
                next_count += 1
            if current_inside:
                out[next_count, 0] = current_s
                out[next_count, 1] = current_m
                next_count += 1
            previous_s, previous_m = current_s, current_m
            previous_value = current_value
            previous_inside = current_inside
        count = next_count
        active = 1 - active
    clipped = _area(first if active == 0 else second, count)
    return min(1.0, max(0.0, clipped / denominator))


@njit
def _first_cell(x, origin, step, n):
    value = np.floor(np.nextafter((x - origin) / step, -np.inf))
    if value <= 0.0:
        return 0
    if value >= n:
        return n
    return int(value)


@njit
def _last_cell(x, origin, step, n):
    value = np.floor(np.nextafter((x - origin) / step, np.inf))
    if value < 0.0:
        return -1
    if value >= n - 1:
        return n - 1
    return int(value)


@njit
def _inverse(s, mu, alpha_perp, alpha_parallel):
    # Angular conversion is independent of s, including when s is zero.
    scale = np.hypot(np.sqrt(max(0.0, 1.0 - mu * mu)) / alpha_perp,
                     mu / alpha_parallel)
    return s * scale, (mu / alpha_parallel) / scale


@njit
def _candidate_bounds(bounds, alpha_perp, alpha_parallel, ns, nm,
                      s_bounds, mu_bounds, q, ds, dm):
    # Inverse-map all four target corners: BL, BR, TL, TR.
    inv_s = np.empty(4, dtype=np.float64)
    inv_mu = np.empty(4, dtype=np.float64)
    target_s = (bounds[0], bounds[1], bounds[0], bounds[1])
    target_mu = (bounds[2], bounds[2], bounds[3], bounds[3])
    for k in range(4):
        inv_s[k], inv_mu[k] = _inverse(target_s[k], target_mu[k], alpha_perp, alpha_parallel)
    slo = np.min(inv_s)
    shi = np.max(inv_s)
    ulo = np.min(inv_mu)
    uhi = np.max(inv_mu)
    ml, mh = mu_bounds[0], mu_bounds[1]
    if uhi <= ml or ulo >= mh:
        return 0, -1, 0, -1
    jm = _first_cell(ulo, ml, dm, nm)
    jp = _last_cell(uhi, ml, dm, nm)
    if jp < jm:
        return 0, -1, 0, -1
    # Widen radial candidates using angular-cell endpoints. This conservatively
    # includes cells intersected by the straight polygon chord.
    qlo = min(q[jm], q[jp + 1])
    qhi = max(q[jm], q[jp + 1])
    slo = min(slo, bounds[0] / qhi)
    shi = max(shi, bounds[1] / qlo)
    sl, sh = s_bounds[0], s_bounds[1]
    if shi <= sl or slo >= sh:
        return 0, -1, 0, -1
    return (_first_cell(slo, sl, ds, ns), _last_cell(shi, sl, ds, ns), jm, jp)


@njit
def map_counts(data, alpha_perp, alpha_parallel, target_shape,
               s_bounds, mu_bounds, mapping_bounds):
    """Conservatively map source counts to a target grid using polygon areas.

    ``mapping_bounds`` is ``(slo, shi, mulo, muhi)``. A target rectangle
    touching this region is transformed with AP factors; other rectangles use
    identity geometry.
    """
    if data.ndim != 3 or data.shape[0] < 1 or data.shape[1] < 1 or data.shape[2] < 1:
        raise ValueError("data must have nonempty shape (source_s, source_mu, channels)")
    ns, nm, channels = data.shape
    nt_s, nt_mu = target_shape[0], target_shape[1]
    if nt_s < 1 or nt_mu < 1:
        raise ValueError("target_shape dimensions must be positive")
    if (not np.isfinite(alpha_perp) or not np.isfinite(alpha_parallel)
            or alpha_perp <= 0.0 or alpha_parallel <= 0.0):
        raise ValueError("AP factors must be finite and positive")
    if len(s_bounds) != 2 or len(mu_bounds) != 2 or len(mapping_bounds) != 4:
        raise ValueError("expected 2 grid bounds and 4 mapping bounds")

    sl, sh = s_bounds[0], s_bounds[1]
    ml, mh = mu_bounds[0], mu_bounds[1]
    if not np.all(np.isfinite(data)):
        raise ValueError("Pair counts and grid/mapping bounds must be finite")
    for value in s_bounds:
        if not np.isfinite(value):
            raise ValueError("Grid bounds must be finite")
    for value in mu_bounds:
        if not np.isfinite(value):
            raise ValueError("Grid bounds must be finite")
    for value in mapping_bounds:
        if not np.isfinite(value):
            raise ValueError("Mapping bounds must be finite")
    if not 0.0 <= sl < sh or not 0.0 <= ml < mh <= 1.0:
        raise ValueError("Invalid s/mu grid bounds")
    if mapping_bounds[1] < mapping_bounds[0] or not 0.0 <= mapping_bounds[2] <= mapping_bounds[3] <= 1.0:
        raise ValueError("Invalid AP mapping bounds")
    source_ds = (sh - sl) / ns
    source_dm = (mh - ml) / nm
    target_ds = (sh - sl) / nt_s
    target_dm = (mh - ml) / nt_mu
    q = np.empty(nm + 1, dtype=np.float64)
    v = np.empty(nm + 1, dtype=np.float64)
    for j in range(nm + 1):
        mu = ml + j * source_dm
        q[j] = np.hypot(alpha_perp * np.sqrt(max(0.0, 1.0 - mu * mu)),
                        alpha_parallel * mu)
        if q[j] <= 0.0 or not np.isfinite(q[j]) or not np.isfinite(sh * q[j]):
            raise ValueError("Invalid AP mapping geometry")
        v[j] = alpha_parallel * mu / q[j]

    for mu in (ml, mh):
        inv_s, inv_mu = _inverse(sh, mu, alpha_perp, alpha_parallel)
        if not np.isfinite(inv_s) or not np.isfinite(inv_mu):
            raise ValueError("Invalid inverse AP mapping geometry")

    output = np.zeros((nt_s, nt_mu, channels), dtype=np.float64)
    identity_q = np.ones(nm + 1, dtype=np.float64)
    identity_v = np.empty(nm + 1, dtype=np.float64)
    for k in range(nm + 1):
        identity_v[k] = ml + k * source_dm
    for i in range(nt_s):
        target_s0 = sl + i * target_ds
        target_s1 = sl + (i + 1) * target_ds
        for j in range(nt_mu):
            target_m0 = ml + j * target_dm
            target_m1 = ml + (j + 1) * target_dm
            bounds = np.array((target_s0, target_s1, target_m0, target_m1))
            # Inclusive overlap means even a target edge on the requested
            # mapping boundary selects transformed geometry.
            use_ap = (target_s1 >= mapping_bounds[0] and target_s0 <= mapping_bounds[1]
                      and target_m1 >= mapping_bounds[2] and target_m0 <= mapping_bounds[3])
            if use_ap:
                aperp, apar, geometry_q, geometry_v = alpha_perp, alpha_parallel, q, v
            else:
                aperp, apar, geometry_q, geometry_v = 1.0, 1.0, identity_q, identity_v
            ci0, ci1, cj0, cj1 = _candidate_bounds(
                bounds, aperp, apar, ns, nm, s_bounds, mu_bounds,
                geometry_q, source_ds, source_dm)
            for si in range(ci0, ci1 + 1):
                s0 = sl + si * source_ds
                s1 = sl + (si + 1) * source_ds
                for mj in range(cj0, cj1 + 1):
                    corners = np.empty((4, 2), dtype=np.float64)
                    corners[0, 0], corners[0, 1] = s0 * geometry_q[mj], geometry_v[mj]
                    corners[1, 0], corners[1, 1] = s1 * geometry_q[mj], geometry_v[mj]
                    corners[2, 0], corners[2, 1] = s0 * geometry_q[mj + 1], geometry_v[mj + 1]
                    corners[3, 0], corners[3, 1] = s1 * geometry_q[mj + 1], geometry_v[mj + 1]
                    rate = polygon_rate(corners, bounds)
                    if rate > 0.0:
                        for c in range(channels):
                            output[i, j, c] += rate * data[si, mj, c]
    return output
