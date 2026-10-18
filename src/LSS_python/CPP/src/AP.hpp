#pragma once

#include "AP_polygon.hpp"
#include "AP_curved.hpp"
#include <cstddef>

namespace lss_ap {

enum class ConvertAlgorithm { Polygon, CurvedGauss8, CurvedAnalytic };

// The numerical core depends only on AP scale ratios and grid geometry.
inline void map_counts(const double* source, double* output, int ns, int nm,
                       int nt_s, int nt_mu, std::ptrdiff_t channels,
                       double alpha_perp, double alpha_parallel,
                       std::array<double,2> sb, std::array<double,2> mb,
                       std::array<double,4> window, int nthreads,
                       ConvertAlgorithm algorithm = ConvertAlgorithm::Polygon) {
    if (algorithm != ConvertAlgorithm::Polygon) {
        map_counts_curved(source, output, ns, nm, nt_s, nt_mu, channels,
                          alpha_perp, alpha_parallel, sb, mb, window, nthreads,
                          algorithm == ConvertAlgorithm::CurvedAnalytic);
        return;
    }
    const Geometry transformed(alpha_perp, alpha_parallel, ns, nm, sb, mb);
    const Geometry identity(1., 1., ns, nm, sb, mb);
    const double ds=(sb[1]-sb[0])/nt_s, dm=(mb[1]-mb[0])/nt_mu;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i=0; i<nt_s; ++i) {
        const double slow=sb[0]+i*ds, shigh=sb[0]+(i+1)*ds;
        for (int j=0; j<nt_mu; ++j) {
            const std::array<double,4> bounds={slow, shigh, mb[0]+j*dm, mb[0]+(j+1)*dm};
            const bool active=shigh>=window[0] && slow<=window[1] &&
                              bounds[3]>=window[2] && bounds[2]<=window[3];
            const auto& geometry=active ? transformed : identity;
            const auto candidate=geometry.candidates(bounds).indices;
            const auto outbase=(std::ptrdiff_t(i)*nt_mu+j)*channels;
            for (int si=candidate[0]; si<=candidate[1]; ++si) {
                for (int mi=candidate[2]; mi<=candidate[3]; ++mi) {
                    double corners[4][2];
                    geometry.corners(si,mi,corners);
                    const double rate=polygon_rate(corners,bounds);
                    if (!(rate>0)) continue;
                    const auto inbase=(std::ptrdiff_t(si)*nm+mi)*channels;
                    for (std::ptrdiff_t c=0; c<channels; ++c)
                        output[outbase+c]+=rate*source[inbase+c];
                }
            }
        }
    }
}

} // namespace lss_ap
