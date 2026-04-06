import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 

def tpcf_comparison(xismu_dicts_list, snaps_list, colors_list, snap_diff_index=(0,1), figsize=(12,12)):
    fig = plt.figure(dpi=100, figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    axes_s = [fig.add_subplot(gs[0, i]) for i in range(2)]
    axes_mu = [fig.add_subplot(gs[1, i]) for i in range(2)]
    ax_diff = fig.add_subplot(gs[2, :])

    colors = colors_list

    for i, (xismu_dict, snaps) in enumerate(zip(xismu_dicts_list, snaps_list)):
        xi_mus = []
        for j, snap in enumerate(snaps):
            xismu_temp = xismu_dict[snap]
            s, xi_s = xismu_temp.integrate_tpcf(s_xis=True, smin=0.0, smax=150.0, is_norm=False, mupack=6)
            axes_s[j].plot(s, xi_s, color=colors[i])
            mu, xi_mu = xismu_temp.integrate_tpcf(intximu=True, smin=6.0, smax=40.0, is_norm=True, mupack=6)
            axes_mu[j].plot(mu, xi_mu, color=colors[i])
            xi_mus.append(xi_mu)
        
        snap1_index, snap2_index = snap_diff_index
        xi_mu_diff = xi_mus[snap1_index] - xi_mus[snap2_index]
        ax_diff.plot(mu, xi_mu_diff, color=colors[i])
    return fig