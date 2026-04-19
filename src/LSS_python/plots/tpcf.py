import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 

def tpcf_comparison(xismu_dicts_list, snaps_list, colors_list, snap_diff_index=(0,1), figsize=(12,12), **argv):
    fig = plt.figure(dpi=100, figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    axes_s = [fig.add_subplot(gs[0, i]) for i in range(2)]
    axes_mu = [fig.add_subplot(gs[1, i]) for i in range(2)]
    ax_diff = fig.add_subplot(gs[2, :])

    colors = colors_list

    s_smin = argv.get("s_smin", 0.0)
    s_smax = argv.get("s_smax", 150.0)
    mu_smin = argv.get("mu_smin", 6.0)
    mu_smax = argv.get("mu_smax", 40.0)
    mupack = argv.get("mupack", 6)
    mumax = argv.get("mumax", 0.97)
    need_slice = argv.get("need_slice", slice(None, None, None))

    for i, (xismu_dict, snaps) in enumerate(zip(xismu_dicts_list, snaps_list)):
        xi_mus = []
        for j, snap in enumerate(snaps):
            xismu_temp = xismu_dict[snap]
            s, xi_s = xismu_temp.integrate_tpcf(s_xis=True, smin=s_smin, smax=s_smax, is_norm=False, mupack=mupack, mumax=mumax)
            axes_s[j].plot(s, xi_s, color=colors[i])
            mu, xi_mu = xismu_temp.integrate_tpcf(intximu=True, smin=mu_smin, smax=mu_smax, is_norm=True, mupack=mupack, mumax=mumax)
            axes_mu[j].plot(mu, xi_mu, color=colors[i])
            xi_mus.append(xi_mu)
        
        snap1_index, snap2_index = snap_diff_index
        xi_mu_diff = xi_mus[snap1_index] - xi_mus[snap2_index]
        ax_diff.plot(mu[need_slice], xi_mu_diff[need_slice], color=colors[i])
    return fig

def test(a, b):
    return 