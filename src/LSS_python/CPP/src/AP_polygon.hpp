#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lss_ap {
struct Point { double s, mu; };

inline double area(const Point* p, int n) {
    if (n<3) return 0.;
    double sum=0.;
    for (int i=1;i+1<n;++i)
        sum+=(p[i].s-p[0].s)*(p[i+1].mu-p[0].mu)
             -(p[i+1].s-p[0].s)*(p[i].mu-p[0].mu);
    return .5*std::abs(sum);
}

inline double polygon_rate(const double corners[4][2], const std::array<double,4>& b) {
    Point buffers[2][12];
    constexpr int order[4]={0,1,3,2};
    double smin=corners[0][0],smax=smin,mmin=corners[0][1],mmax=mmin;
    for (int i=0;i<4;++i) {
        buffers[0][i]={corners[order[i]][0],corners[order[i]][1]};
        smin=std::min(smin,corners[i][0]);smax=std::max(smax,corners[i][0]);
        mmin=std::min(mmin,corners[i][1]);mmax=std::max(mmax,corners[i][1]);
    }
    if (smax<=b[0] || smin>=b[1] || mmax<=b[2] || mmin>=b[3]) return 0.;
    const double denominator=area(buffers[0],4);
    if (!(denominator>0)) return 0.;
    if (smin>=b[0] && smax<=b[1] && mmin>=b[2] && mmax<=b[3]) return 1.;
    int active=0,count=4;
    for (int edge=0;edge<4 && count;++edge) {
        Point* input=buffers[active];Point* output=buffers[1-active];
        const int axis=edge/2;const bool above=edge%2==0;const double bound=b[edge];
        const auto value=[&](const Point& p){return axis==0?p.s:p.mu;};
        const auto inside=[&](const Point& p){return above?value(p)>=bound:value(p)<=bound;};
        Point previous=input[count-1];bool previous_inside=inside(previous);int next=0;
        for (int i=0;i<count;++i) {
            const Point current=input[i];const bool current_inside=inside(current);
            if (previous_inside!=current_inside) {
                const double t=(bound-value(previous))/(value(current)-value(previous));
                output[next++]={previous.s+t*(current.s-previous.s),previous.mu+t*(current.mu-previous.mu)};
            }
            if (current_inside) output[next++]=current;
            previous=current;previous_inside=current_inside;
        }
        count=next;active=1-active;
    }
    return std::clamp(area(buffers[active],count)/denominator,0.,1.);
}

struct Candidates {
    // Inclusive cell indices; high < low means there is no overlap.
    std::array<int,4> indices={0,-1,0,-1};
    std::array<Point,4> inverse;
};

class Geometry {
public:
    int ns,nm;
    double a,b,sl,sh,ml,mh,ds,dm;
    std::vector<double> q,v;

    Geometry(double aperp,double apar,int n_s,int n_mu,
             std::array<double,2> sb,std::array<double,2> mb)
        :ns(n_s),nm(n_mu),a(aperp),b(apar),sl(sb[0]),sh(sb[1]),ml(mb[0]),mh(mb[1]),
         ds((sh-sl)/ns),dm((mh-ml)/nm),q(nm+1),v(nm+1) {
        for (int j=0;j<=nm;++j) {
            const double u=ml+j*dm;
            q[j]=std::hypot(a*std::sqrt(std::max(0.,1-u*u)),b*u);
            v[j]=b*u/q[j];
            if (!(q[j]>0) || !std::isfinite(q[j]) || !std::isfinite(sh*q[j]))
                throw std::runtime_error("Invalid AP coordinate transform: non-finite mapped geometry");
        }
    }

    Point inverse(double s,double mu) const {
        const double scale=std::hypot(std::sqrt(std::max(0.,1-mu*mu))/a,mu/b);
        // The angular transform is independent of s, including at s=0.
        return {s*scale,(mu/b)/scale};
    }

    int first_cell(double x,double origin,double step,int n) const {
        const double value=std::floor(std::nextafter((x-origin)/step,-std::numeric_limits<double>::infinity()));
        return value<=0?0:(value>=n?n:static_cast<int>(value));
    }
    int last_cell(double x,double origin,double step,int n) const {
        const double value=std::floor(std::nextafter((x-origin)/step,std::numeric_limits<double>::infinity()));
        return value<0?-1:(value>=n-1?n-1:static_cast<int>(value));
    }

    Candidates candidates(const std::array<double,4>& bounds) const {
        Candidates result;
        // AP corner order: bottom left, bottom right, top left, top right.
        result.inverse={inverse(bounds[0],bounds[2]),inverse(bounds[1],bounds[2]),
                        inverse(bounds[0],bounds[3]),inverse(bounds[1],bounds[3])};
        double s_lo=result.inverse[0].s,s_hi=s_lo,u_lo=result.inverse[0].mu,u_hi=u_lo;
        for (const auto& p:result.inverse) {
            s_lo=std::min(s_lo,p.s);s_hi=std::max(s_hi,p.s);
            u_lo=std::min(u_lo,p.mu);u_hi=std::max(u_hi,p.mu);
        }
        if (u_hi<=ml || u_lo>=mh) return result;
        const int jm=first_cell(u_lo,ml,dm,nm),jp=last_cell(u_hi,ml,dm,nm);
        if (jp<jm) return result;
        // Include the angular-cell endpoints as well as the inverse corners.
        // A chord can intersect a target even when the exact curved edge does not.
        const double qlo=std::min(q[jm],q[jp+1]),qhi=std::max(q[jm],q[jp+1]);
        s_lo=std::min(s_lo,bounds[0]/qhi);
        s_hi=std::max(s_hi,bounds[1]/qlo);
        if (s_hi<=sl || s_lo>=sh) return result;
        result.indices={first_cell(s_lo,sl,ds,ns),last_cell(s_hi,sl,ds,ns),jm,jp};
        return result;
    }

    void corners(int i,int j,double points[4][2]) const {
        const double s0=sl+i*ds,s1=sl+(i+1)*ds;
        points[0][0]=s0*q[j];points[0][1]=v[j];
        points[1][0]=s1*q[j];points[1][1]=v[j];
        points[2][0]=s0*q[j+1];points[2][1]=v[j+1];
        points[3][0]=s1*q[j+1];points[3][1]=v[j+1];
    }
};
} // namespace lss_ap
