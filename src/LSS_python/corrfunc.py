from Corrfunc.theory import DDsmu


def call_DDsmu(data1, data2, sedges, mubin, with_weight, boxsize,
                refine_factors=(2, 2, 1), nthreads=1, autocorr=True, verbose=False):
    """
    Wrapper for DDsmu call.
    """
    x_refine_factor, y_refine_factor, z_refine_factor = refine_factors

    if with_weight:
        if autocorr:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges,
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                weights1=data1[:, 3], weight_type="pair_product",
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )
        else:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges,
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                weights1=data1[:, 3], weight_type="pair_product",
                X2=data2[:, 0], Y2=data2[:, 1], Z2=data2[:, 2],
                weights2=data2[:, 3],
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )
    else:
        if autocorr:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges,
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )
        else:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges,
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                X2=data2[:, 0], Y2=data2[:, 1], Z2=data2[:, 2],
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )

    return result
