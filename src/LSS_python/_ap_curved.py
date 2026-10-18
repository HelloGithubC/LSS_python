"""Exact curved AP cell geometry with adaptive Gauss8 or analytic integrals."""

import numpy as np
from numba import njit


@njit
def radius(v, a, b):
    if a == b:
        if a <= 0.0 or not np.isfinite(a):
            raise RuntimeError("curved AP radius is not representable")
        return a
    inverse = np.hypot(np.sqrt(max(0.0, (1.0-v)*(1.0+v))) / a, v / b)
    if inverse <= 0.0 or not np.isfinite(inverse):
        raise RuntimeError("curved AP radius is not representable")
    value = 1.0/inverse
    if not np.isfinite(value):
        raise RuntimeError("curved AP radius is not representable")
    return value


@njit
def _curved_root(wanted, a, b):
    if a == b or wanted <= 0.0 or not np.isfinite(wanted):
        return np.nan
    chi = ((a-b)/b)*(a/b+1.0)
    ar = a/wanted
    square = ((a-wanted)/wanted)*(ar+1.0)/chi
    if not 0.0 < square < 1.0:
        return np.nan
    return np.sqrt(square)


@njit
def integral_R(lo, hi, a, b):
    """Stable definite integral, including nearly isotropic and narrow bins."""
    if hi <= lo:
        return 0.0
    if a == b:
        return a*(hi-lo)
    chi = ((a-b)/b) * (a/b+1.0)
    width = hi-lo
    if abs(chi) < 1e-8:
        h2, l2 = hi*hi, lo*lo
        p2 = h2 + hi*lo + l2
        p4 = h2*h2 + h2*hi*lo + h2*l2 + hi*lo*l2 + l2*l2
        return a*width*(1.0-chi*p2/6.0 + 3.0*chi*chi*p4/40.0)
    k = np.sqrt(abs(chi))
    x0, x1 = k*lo, k*hi
    dx = k*width
    if chi > 0.0:
        den = x1*np.hypot(1.0, x0) + x0*np.hypot(1.0, x1)
        delta = np.arcsinh(dx*(x1+x0)/den) if den > 0.0 else np.arcsinh(x1)-np.arcsinh(x0)
    else:
        x0, x1 = min(1.0, max(0.0, x0)), min(1.0, max(0.0, x1))
        y0 = np.sqrt(max(0.0, (1.0-x0)*(1.0+x0)))
        y1 = np.sqrt(max(0.0, (1.0-x1)*(1.0+x1)))
        den = x1*y0+x0*y1
        sine = dx*(x1+x0)/den if den > 0.0 else x1*y0-x0*y1
        delta = np.arctan2(sine, y0*y1+x0*x1)
    return a*delta/k


@njit
def _width(v, sl, sh, s0, s1, a, b):
    r = radius(v, a, b)
    if s0*r >= sl and s1*r <= sh:
        return (s1-s0)*r
    if sl >= s0*r and sh <= s1*r:
        return sh-sl
    return max(0.0, min(sh, s1*r)-max(sl, s0*r))


@njit
def _gauss8(lo, hi, sl, sh, s0, s1, a, b, integrate_radius):
    nodes = (0.18343464249564980494, 0.52553240991632898582,
             0.79666647741362673959, 0.96028985649753623168)
    weights = (0.36268378337836198297, 0.31370664587788728734,
               0.22238103445337447054, 0.10122853629037625915)
    half = (hi-lo)*0.5
    mid = lo+half
    total = 0.0
    for i in range(4):
        v0, v1 = mid-half*nodes[i], mid+half*nodes[i]
        if integrate_radius:
            total += weights[i]*(radius(v0, a, b)+radius(v1, a, b))
        else:
            total += weights[i]*(_width(v0, sl, sh, s0, s1, a, b)
                                 + _width(v1, sl, sh, s0, s1, a, b))
    return half*total


@njit
def _integration_scale(lo, hi, sl, sh, s0, s1, a, b, integrate_radius):
    # Retain the midpoint sample as well as the mathematically sufficient
    # endpoints: rounded evaluations need not preserve exact monotonicity.
    mid = lo+(hi-lo)*0.5
    qmax = max(radius(lo, a, b), radius(hi, a, b), radius(mid, a, b))
    if integrate_radius:
        return qmax
    return max(abs(s0*qmax), abs(s1*qmax), abs(sl), abs(sh))


@njit
def adaptive_R_or_width(lo, hi, sl, sh, s0, s1, a, b, integrate_radius):
    """Reuse estimates with the same left-first stack and sum in both backends."""
    if hi <= lo:
        return 0.0
    if a == b:
        value = a if integrate_radius else _width(lo, sl, sh, s0, s1, a, b)
        result = (hi-lo)*value
        if not np.isfinite(result):
            raise RuntimeError("curved_gauss8 integral is not finite")
        return result
    lows = np.empty(26)
    highs = np.empty(26)
    wholes = np.empty(26)
    depths = np.empty(26, dtype=np.int64)
    lows[0], highs[0], depths[0] = lo, hi, 0
    wholes[0] = _gauss8(lo, hi, sl, sh, s0, s1, a, b, integrate_radius)
    top, result = 0, 0.0
    while top >= 0:
        low, high, whole, depth = lows[top], highs[top], wholes[top], depths[top]
        top -= 1
        mid = low+(high-low)*0.5
        scale = _integration_scale(low, high, sl, sh, s0, s1, a, b, integrate_radius)
        noise = 16.0*np.finfo(np.float64).eps*scale*(high-low)
        if not np.isfinite(whole):
            raise RuntimeError("curved_gauss8 integral is not finite")
        if mid <= low or mid >= high:
            # No representable interior exists. Bound the variation of this
            # smooth, monotone piece instead of claiming a false subdivision.
            f0 = radius(low, a, b) if integrate_radius else _width(low, sl, sh, s0, s1, a, b)
            f1 = radius(high, a, b) if integrate_radius else _width(high, sl, sh, s0, s1, a, b)
            if abs(f1-f0)*(high-low) <= max(1e-12*abs(whole), noise):
                result += whole
                continue
            raise RuntimeError("curved_gauss8 interval cannot be bisected")
        left = _gauss8(low, mid, sl, sh, s0, s1, a, b, integrate_radius)
        right = _gauss8(mid, high, sl, sh, s0, s1, a, b, integrate_radius)
        split = left+right
        if not np.isfinite(split):
            raise RuntimeError("curved_gauss8 integral is not finite")
        if abs(split-whole) <= max(1e-12*abs(split), noise):
            result += split
            continue
        if depth >= 24:
            raise RuntimeError("curved_gauss8 failed to converge at depth 24")
        # Push right first to preserve left-to-right summation.
        top += 1
        lows[top], highs[top], wholes[top], depths[top] = mid, high, right, depth+1
        top += 1
        lows[top], highs[top], wholes[top], depths[top] = low, mid, left, depth+1
    if not np.isfinite(result):
        raise RuntimeError("curved_gauss8 integral is not finite")
    return result


@njit
def source_areas(v, source_ds, a, b, algorithm):
    areas = np.empty(len(v)-1)
    for j in range(len(areas)):
        if a == b:
            value = a*(v[j+1]-v[j])
        elif algorithm == 1:
            value = adaptive_R_or_width(v[j], v[j+1], 0.0, 0.0, 0.0, 0.0, a, b, True)
        else:
            value = integral_R(v[j], v[j+1], a, b)
        areas[j] = source_ds*value
        if not np.isfinite(areas[j]) or areas[j] <= 0.0:
            raise RuntimeError("curved AP source area is not representable")
    return areas


@njit
def curved_rate(s0, s1, v0, v1, denominator, bounds, a, b, algorithm):
    lo, hi = max(v0, bounds[2]), min(v1, bounds[3])
    if hi <= lo:
        return 0.0
    qmin = min(radius(lo, a, b), radius(hi, a, b))
    qmax = max(radius(lo, a, b), radius(hi, a, b))
    sl, sh = bounds[0], bounds[1]
    if s1*qmax <= sl or s0*qmin >= sh:
        return 0.0
    if lo == v0 and hi == v1 and s0*qmin >= sl and s1*qmax <= sh:
        return 1.0
    area = curved_overlap(lo, hi, sl, sh, s0, s1, a, b, algorithm == 2)
    rate = area/denominator
    if not np.isfinite(rate):
        raise RuntimeError("curved AP overlap area is not finite")
    if rate < -1e-10 or rate > 1.0+1e-10:
        raise RuntimeError("curved AP overlap rate lies outside [0, 1]")
    return min(1.0, max(0.0, rate))


@njit
def curved_overlap(lo, hi, sl, sh, s0, s1, a, b, analytic):
    """Shared cuts and segment decisions; only the integral strategy differs."""
    if hi <= lo or s1 <= s0 or sh <= sl:
        return 0.0
    if a == b:
        area = (hi-lo)*_width(lo, sl, sh, s0, s1, a, b)
        if not np.isfinite(area):
            raise RuntimeError("curved AP overlap area is not finite")
        return area
    cuts = np.empty(6)
    cuts[0], cuts[1] = lo, hi
    count = 2
    for target in (sl, sh):
        for source in (s0, s1):
            if target <= 0.0 or source <= 0.0:
                continue
            root = _curved_root(target/source, a, b)
            if np.isfinite(root) and lo < root < hi:
                cuts[count] = root
                count += 1
    # Six elements at most; insertion sorting preserves repeated intersections.
    for i in range(1, count):
        x, j = cuts[i], i-1
        while j >= 0 and cuts[j] > x:
            cuts[j+1] = cuts[j]
            j -= 1
        cuts[j+1] = x
    area = 0.0
    for k in range(count-1):
        low, high = cuts[k], cuts[k+1]
        if high <= low:
            continue
        mid = low+(high-low)*0.5
        r = radius(mid, a, b)
        upper, lower = min(sh, s1*r), max(sl, s0*r)
        # All radial crossings are already cuts, so the active width formula
        # cannot change inside a segment. The midpoint classifies its branch.
        if upper <= lower:
            continue
        if not analytic:
            piece = adaptive_R_or_width(low, high, sl, sh, s0, s1, a, b, False)
        else:
            c = (0.0 if sh < s1*r else s1) - (0.0 if sl > s0*r else s0)
            d = (sh if sh < s1*r else 0.0) - (sl if sl > s0*r else 0.0)
            piece = c*integral_R(low, high, a, b)+d*(high-low)
        area += piece
    if not np.isfinite(area):
        raise RuntimeError("curved AP overlap area is not finite")
    return area
