import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 

def fftpower_comparison(fftpower_dicts_list, snaps_list, colors_list, figsize=(12,12), **argv):
    fig = plt.figure(dpi=100, figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    axes_k = [fig.add_subplot(gs[0, i]) for i in range(2)]
    axes_mu = [fig.add_subplot(gs[1, i]) for i in range(2)]
    ax_diff = fig.add_subplot(gs[2, :])

    colors = colors_list

    k_kmin = argv.get("k_kmin", 0.1)
    k_kmax = argv.get("k_kmax", 1.5)
    mu_kmin = argv.get("mu_kmin", 0.3)
    mu_kmax = argv.get("mu_kmax", 0.8)
    mu_max = argv.get("mu_max", -1.0)

    if isinstance(fftpower_dicts_list, dict):
        fftpower_dicts_list = [fftpower_dicts_list, ]
    if not isinstance(snaps_list[0], tuple) and not isinstance(snaps_list[0], list):
        snaps_list = [snaps_list, ]

    for i, (fftpower_dict, snaps) in enumerate(zip(fftpower_dicts_list, snaps_list)):
        Pmus = []
        if len(snaps) != 2:
            raise ValueError("The length of snaps (the element of snaps_list) should be 2.")
        for j, snap in enumerate(snaps):
            fftpower_temp = fftpower_dict[snap]
            k, Pk = fftpower_temp.intergrate_fftpower(k_min=k_kmin, k_max=k_kmax, mu_min=-1.0, mu_max=-1.0, integrate="mu", norm=False)
            axes_k[j].loglog(k, Pk, color=colors[i])
            mu, Pmu = fftpower_temp.intergrate_fftpower(k_min=mu_kmin, k_max=mu_kmax, mu_min=-1.0, mu_max=mu_max, integrate="k", norm=True)
            axes_mu[j].plot(mu, Pmu, color=colors[i])
            Pmus.append(Pmu)
        
        Pmu_diff = Pmus[0] - Pmus[1]
        ax_diff.plot(mu, Pmu_diff, color=colors[i])
    return fig