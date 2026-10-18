#pragma once

#include "AP_polygon.hpp"
#include <exception>
#include <string>

namespace lss_ap {

inline double curved_R(double v, double a, double b) {
    if (a == b) {
        if (!(a > 0.) || !std::isfinite(a))
            throw std::runtime_error("curved AP radius is not representable");
        return a;
    }
    const double inverse = std::hypot(std::sqrt(std::max(0., (1. - v) * (1. + v))) / a, v / b);
    if (!(inverse > 0.) || !std::isfinite(inverse))
        throw std::runtime_error("curved AP radius is not representable");
    const double value = 1. / inverse;
    if (!(value > 0.) || !std::isfinite(value))
        throw std::runtime_error("curved AP radius is not representable");
    return value;
}

inline double curved_root(double radius, double a, double b) {
    if (a == b) return std::numeric_limits<double>::quiet_NaN();
    if (!(radius > 0.) || !std::isfinite(radius)) return std::numeric_limits<double>::quiet_NaN();
    const double chi = ((a - b) / b) * (a / b + 1.);
    const double ar = a / radius;
    const double y = ((a - radius) / radius) * (ar + 1.) / chi;
    if (!(y > 0. && y < 1.)) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(y);
}

inline void curved_sort_cuts(double* cuts, int n) {
    for (int i = 1; i < n; ++i) {
        const double value = cuts[i];
        int j = i;
        while (j > 0 && cuts[j - 1] > value) {
            cuts[j] = cuts[j - 1];
            --j;
        }
        cuts[j] = value;
    }
}

inline double curved_int_R(double lo, double hi, double a, double b) {
    if (!(hi > lo)) return 0.;
    if (a == b) return a * (hi - lo);
    const double chi = ((a - b) / b) * (a / b + 1.);
    if (std::abs(chi) < 1.e-8) {
        const double h2 = hi * hi, l2 = lo * lo;
        const double second = h2 + hi * lo + l2;
        const double fourth = h2 * h2 + h2 * hi * lo + h2 * l2 + hi * lo * l2 + l2 * l2;
        return a * (hi - lo) * (1. - chi * second / 6. + 3. * chi * chi * fourth / 40.);
    }
    if (chi > 0.) {
        const double k = std::sqrt(chi), x0 = k * lo, dx = k * (hi - lo), x1 = k * hi;
        const double y0 = std::hypot(1., x0), y1 = std::hypot(1., x1);
        const double numerator = dx * (x1 + x0);
        const double denominator = x1 * y0 + x0 * y1;
        const double delta = !(denominator > 0.) ? std::asinh(x1) - std::asinh(x0)
                                               : std::asinh(numerator / denominator);
        return a * delta / k;
    }
    const double rootchi = std::sqrt(-chi);
    const double x0 = std::clamp(rootchi * lo, 0., 1.);
    const double dx = rootchi * (hi - lo);
    const double x1 = std::clamp(rootchi * hi, 0., 1.);
    const double y0 = std::sqrt(std::max(0., (1. - x0) * (1. + x0)));
    const double y1 = std::sqrt(std::max(0., (1. - x1) * (1. + x1)));
    const double den = x1 * y0 + x0 * y1;
    const double delta = !(den > 0.) ? std::atan2(y0 * x1 - x0 * y1, y1 * y0 + x0 * x1)
                                   : std::atan2(dx * (x1 + x0) / den,
                                                y1 * y0 + x0 * x1);
    return a * delta / rootchi;
}

inline double curved_width(double v, double sl, double sh, double s0, double s1,
                           double a, double b) {
    const double r = curved_R(v, a, b);
    if (s0 * r >= sl && s1 * r <= sh) return (s1 - s0) * r;
    if (sl >= s0 * r && sh <= s1 * r) return sh - sl;
    return std::max(0., std::min(sh, s1 * r) - std::max(sl, s0 * r));
}

template<bool IntegrateRadius>
inline double curved_gauss8_once(double lo, double hi, double sl, double sh,
                                 double s0, double s1, double a, double b) {
    static constexpr double x[4] = {0.18343464249564980494, 0.52553240991632898582,
                                    0.79666647741362673959, 0.96028985649753623168};
    static constexpr double w[4] = {0.36268378337836198297, 0.31370664587788728734,
                                    0.22238103445337447054, 0.10122853629037625915};
    const double half = (hi - lo) * .5, mid = lo + half;
    double sum = 0.;
    for (int i = 0; i < 4; ++i) {
        const double v0 = mid - half * x[i], v1 = mid + half * x[i];
        if constexpr (IntegrateRadius)
            sum += w[i] * (curved_R(v0, a, b) + curved_R(v1, a, b));
        else
            sum += w[i] * (curved_width(v0, sl, sh, s0, s1, a, b) +
                           curved_width(v1, sl, sh, s0, s1, a, b));
    }
    return half * sum;
}

template<bool IntegrateRadius>
inline double curved_integration_scale(double lo, double hi, double sl, double sh,
                                       double s0, double s1, double a, double b) {
    // Rounded evaluations need not preserve exact mathematical monotonicity.
    const double mid = lo + (hi - lo) * .5;
    const double qmax = std::max({curved_R(lo, a, b), curved_R(hi, a, b), curved_R(mid, a, b)});
    if constexpr (IntegrateRadius) return qmax;
    return std::max({std::abs(s0 * qmax), std::abs(s1 * qmax), std::abs(sl), std::abs(sh)});
}

template<bool IntegrateRadius>
inline double curved_adaptive_integral(double lo, double hi, double sl, double sh,
                                       double s0, double s1, double a, double b) {
    if (!(hi > lo)) return 0.;
    if (a == b) {
        const double value = IntegrateRadius ? a : curved_width(lo, sl, sh, s0, s1, a, b);
        const double result = (hi - lo) * value;
        if (!std::isfinite(result)) throw std::runtime_error("curved_gauss8 integral is not finite");
        return result;
    }
    // Identical left-first depth-first walk and leaf summation to Numba.
    // At most depth+1 pending frames exist for maximum recursion depth 24.
    double lows[26], highs[26], wholes[26];
    int depths[26];
    lows[0] = lo; highs[0] = hi; depths[0] = 0;
    wholes[0] = curved_gauss8_once<IntegrateRadius>(lo, hi, sl, sh, s0, s1, a, b);
    int top = 0;
    double result = 0.;
    while (top >= 0) {
        const double low = lows[top], high = highs[top], whole = wholes[top];
        const int depth = depths[top--];
        const double mid = low + (high - low) * .5;
        const double scale = curved_integration_scale<IntegrateRadius>(low, high, sl, sh, s0, s1, a, b);
        const double noise = 16. * std::numeric_limits<double>::epsilon() * scale * (high - low);
        if (!std::isfinite(whole)) throw std::runtime_error("curved_gauss8 integral is not finite");
        if (mid <= low || mid >= high) {
            // Bound the variation on this monotone piece when no floating
            // interior point exists; do not claim a false subdivision.
            const double f0 = IntegrateRadius ? curved_R(low, a, b) : curved_width(low, sl, sh, s0, s1, a, b);
            const double f1 = IntegrateRadius ? curved_R(high, a, b) : curved_width(high, sl, sh, s0, s1, a, b);
            if (std::abs(f1 - f0) * (high - low) <= std::max(1.e-12 * std::abs(whole), noise)) {
                result += whole;
                continue;
            }
            throw std::runtime_error("curved_gauss8 interval cannot be bisected");
        }
        const double left = curved_gauss8_once<IntegrateRadius>(low, mid, sl, sh, s0, s1, a, b);
        const double right = curved_gauss8_once<IntegrateRadius>(mid, high, sl, sh, s0, s1, a, b);
        const double split = left + right;
        if (!std::isfinite(split)) throw std::runtime_error("curved_gauss8 integral is not finite");
        if (std::abs(split - whole) <= std::max(1.e-12 * std::abs(split), noise)) {
            result += split;
            continue;
        }
        if (depth >= 24) throw std::runtime_error("curved_gauss8 failed to converge at depth 24");
        ++top; lows[top] = mid; highs[top] = high; wholes[top] = right; depths[top] = depth + 1;
        ++top; lows[top] = low; highs[top] = mid; wholes[top] = left; depths[top] = depth + 1;
    }
    if (!std::isfinite(result)) throw std::runtime_error("curved_gauss8 integral is not finite");
    return result;
}

inline double curved_overlap(double lo, double hi, double sl, double sh,
                             double s0, double s1, double a, double b, bool analytic) {
    if (!(hi > lo) || !(s1 > s0) || !(sh > sl)) return 0.;
    if (a == b) {
        const double area = (hi - lo) * curved_width(lo, sl, sh, s0, s1, a, b);
        if (!std::isfinite(area)) throw std::runtime_error("curved AP overlap area is not finite");
        return area;
    }
    double cuts[6] = {lo, hi, 0., 0., 0., 0.};
    int ncuts = 2;
    for (double edge : {sl, sh}) for (double radial : {s0, s1}) {
        if (!(edge > 0.) || !(radial > 0.)) continue;
        const double root = curved_root(edge / radial, a, b);
        if (std::isfinite(root) && root > lo && root < hi) cuts[ncuts++] = root;
    }
    curved_sort_cuts(cuts, ncuts);
    double area = 0.;
    for (int k = 0; k + 1 < ncuts; ++k) {
        const double low = cuts[k], high = cuts[k + 1];
        if (!(high > low)) continue;
        const double mid = low + (high - low) * .5, r = curved_R(mid, a, b);
        const double upper = std::min(sh, s1 * r), lower = std::max(sl, s0 * r);
        // Every radial crossing is already a cut: the midpoint classifies
        // this segment without introducing an area cutoff tolerance.
        if (upper <= lower) continue;
        double piece;
        if (!analytic) {
            piece = curved_adaptive_integral<false>(low, high, sl, sh, s0, s1, a, b);
        } else {
            const double c = (sh < s1 * r ? 0. : s1) - (sl > s0 * r ? 0. : s0);
            const double d = (sh < s1 * r ? sh : 0.) - (sl > s0 * r ? sl : 0.);
            piece = c * curved_int_R(low, high, a, b) + d * (high - low);
        }
        area += piece;
    }
    if (!std::isfinite(area)) throw std::runtime_error("curved AP overlap area is not finite");
    return area;
}

inline void map_counts_curved(const double* source, double* output, int ns, int nm,
                              int nt_s, int nt_mu, std::ptrdiff_t channels,
                              double a, double b, std::array<double,2> sb,
                              std::array<double,2> mb, std::array<double,4> window,
                              int nthreads, bool analytic) {
    const Geometry geometry(a, b, ns, nm, sb, mb);
    const Geometry identity(1., 1., ns, nm, sb, mb);
    const double source_ds = geometry.ds;
    const double target_ds = (sb[1] - sb[0]) / nt_s, dmu = (mb[1] - mb[0]) / nt_mu;
    std::vector<double> source_area(nm);
    for (int mi = 0; mi < nm; ++mi) {
        const double integral = a == b ? a * (geometry.v[mi + 1] - geometry.v[mi]) : analytic ? curved_int_R(geometry.v[mi], geometry.v[mi + 1], a, b)
                                         : curved_adaptive_integral<true>(geometry.v[mi], geometry.v[mi + 1], 0., 0., 0., 0., a, b);
        source_area[mi] = source_ds * integral;
        if (!(source_area[mi] > 0.) || !std::isfinite(source_area[mi]))
            throw std::runtime_error("curved AP source area is not representable");
    }
    std::vector<std::exception_ptr> errors(std::size_t(nt_s) * nt_mu);
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < nt_s; ++i) {
        const double slow = sb[0] + i * target_ds, shigh = sb[0] + (i + 1) * target_ds;
        for (int j = 0; j < nt_mu; ++j) {
            try {
                const std::array<double,4> bounds = {slow, shigh, mb[0] + j * dmu, mb[0] + (j + 1) * dmu};
                const bool active = shigh >= window[0] && slow <= window[1] &&
                                    bounds[3] >= window[2] && bounds[2] <= window[3];
                const auto& current_geometry = active ? geometry : identity;
                const auto candidate = current_geometry.candidates(bounds).indices;
                const auto outbase = (std::ptrdiff_t(i) * nt_mu + j) * channels;
                for (int si = candidate[0]; si <= candidate[1]; ++si) {
                    for (int mi = candidate[2]; mi <= candidate[3]; ++mi) {
                        double rate;
                        if (!active) {
                            double corners[4][2];
                            identity.corners(si, mi, corners);
                            rate = polygon_rate(corners, bounds);
                        } else {
                            const double s0 = sb[0] + si * source_ds, s1 = sb[0] + (si + 1) * source_ds;
                            const double vl = std::max(geometry.v[mi], bounds[2]);
                            const double vh = std::min(geometry.v[mi + 1], bounds[3]);
                            if (!(vh > vl)) continue;
                            const double qmin = std::min(curved_R(vl, a, b), curved_R(vh, a, b));
                            const double qmax = std::max(curved_R(vl, a, b), curved_R(vh, a, b));
                            if (s1 * qmax <= slow || s0 * qmin >= shigh) continue;
                            if (vl == geometry.v[mi] && vh == geometry.v[mi + 1] &&
                                s0 * qmin >= slow && s1 * qmax <= shigh) {
                                rate = 1.;
                            } else {
                                const double overlap = curved_overlap(vl, vh, slow, shigh, s0, s1, a, b, analytic);
                                if (!std::isfinite(overlap)) throw std::runtime_error("curved AP overlap area is not finite");
                                rate = overlap / source_area[mi];
                                if (!std::isfinite(rate)) throw std::runtime_error("curved AP overlap rate is not finite");
                                if (rate < -1.e-10 || rate > 1. + 1.e-10)
                                    throw std::runtime_error("curved AP overlap rate lies outside [0, 1]");
                                rate = std::clamp(rate, 0., 1.);
                            }
                        }
                        if (!(rate > 0.)) continue;
                        const auto inbase = (std::ptrdiff_t(si) * nm + mi) * channels;
                        for (std::ptrdiff_t c = 0; c < channels; ++c) output[outbase + c] += rate * source[inbase + c];
                    }
                }
            } catch (...) { errors[std::size_t(i) * nt_mu + j] = std::current_exception(); }
        }
    }
    for (const auto& error : errors) if (error) {
        try { std::rethrow_exception(error); }
        catch (const std::exception& e) { throw std::runtime_error(e.what()); }
    }
}

} // namespace lss_ap
