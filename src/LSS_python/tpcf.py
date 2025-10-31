import joblib, os
import numpy as np 
from Corrfunc.theory import DDsmu
from Corrfunc.mocks import DDsmu_mocks

from .base import Hz, DA, comov_dist, traz

def meannorm(X, axis=0):
    if len(X.shape) == 1:
        axis = None
    if np.mean(X) > 0:
        return X / np.mean(X, axis=axis)
    else:
        # The mean value is from -1 to 1
        return X / abs(np.mean(X, axis=axis)) + 2
    
def packarray2d(X, rat, axis=1):
    subCount = int(X.shape[axis] / rat)
    new_shape = (subCount, X.shape[1]) if axis == 0 else (X.shape[0], subCount)
    Xs = np.zeros(new_shape)
    for i in range(subCount):
        if axis == 0:
            Xs[i, :] = np.mean(X[i * rat : (i + 1) * rat], axis=axis)
        else:
            Xs[:, i] = np.mean(X[:,i * rat : (i + 1) * rat], axis=axis)
    return Xs 

def packarray1d(X, rat):
    X_new = X[np.newaxis,:]
    return packarray2d(X_new, rat, axis=1)[0]

class xismu(object):
    def __init__(
        self,
        smax=150, sbin=150, mubin=120,
        DDnorm=None, DRnorm=None, RRnorm=None,
        S=None, Mu=None, 
        DD=None, DR=None, RR=None,
        set_data=True,
    ):
        """
        data_type: CUTE, BINARY(As Corrfunc)
        Note: only support DR instead of D1R2 and D2R1 now.
        filename: filename of data. If None, DD, DR, RR before normalization and other parameters are required.
        """
        self.smax, self.sbin, self.mubin = smax, sbin, mubin
        
        if set_data:
            self.DDnorm, self.DRnorm, self.RRnorm = DDnorm, DRnorm, RRnorm
            self.S = S 
            self.Mu = Mu
            self.DD = DD.reshape((self.sbin, self.mubin))
            self.DR = DR.reshape((self.sbin, self.mubin))
            self.RR = RR.reshape((self.sbin, self.mubin))
            self.DD = self.DD / self.DDnorm
            self.DR = self.DR / self.DRnorm
            self.RR = self.RR / self.RRnorm
            self.RR[self.RR == 0] = 1e-15
        
    @classmethod
    def load(cls, filename, data_type="BINARY", smax=150, sbin=150, mubin=120, deal_with_0s0mu=True):
        """
            deal_with_0s0mu: deal with the case that s=0 and mu=0. Only be valid when using BINARY data.
        """
        self = xismu(smax, sbin, mubin, set_data=False)
        self.filename = filename
        self.data_type = data_type
        if self.data_type == "CUTE":
            self._read_from_CUTE(filename, self.sbin, self.mubin)
        elif self.data_type == "BINARY":
            self._read_from_BINARY(filename, deal_with_0s0mu)
            if self.DD.shape[0] != self.sbin or self.DD.shape[1] != self.mubin:
                raise ValueError("sbin or mubin is not equal to the data")
        else:
            raise ValueError("data_type must be CUTE or BINARY")
        
        self.DD = self.DD / self.DDnorm
        self.DR = self.DR / self.DRnorm
        self.RR = self.RR / self.RRnorm
        self.RR[self.RR == 0.0] = 1e-15

        if self.S is None or self.Mu is None:
            raise ValueError("S and Mu must be included in the data when using load")
        return self

    def save(self, filename, with_weight=True, has_converted=False):
        result_dict = {}
        result_dict["with_weight"] = with_weight
        if with_weight:
            result_dict["DDwpairs"] = self.DD * self.DDnorm 
            result_dict["RRwpairs"] = self.RR * self.RRnorm
            result_dict["DRwpairs"] = self.DR * self.DRnorm
        else:
            result_dict["DDnpairs"] = self.DD * self.DDnorm
            result_dict["RRnpairs"] = self.RR * self.RRnorm
            result_dict["DRnpairs"] = self.DR * self.DRnorm
        # result_dict["tpCF"] = self.xis 
        result_dict["norm_d1d2"] = self.DDnorm 
        result_dict["norm_d1r2"] = self.DRnorm
        result_dict["norm_r1r2"] = self.RRnorm
        
        if has_converted:
            result_dict["s_array"] = self.s_array
            result_dict["mu_array"] = self.mu_array
        else:
            result_dict["s_array"] = self.S[:,0]
            result_dict["mu_array"] = self.Mu[0]
        joblib.dump(result_dict, filename)


    def _read_from_CUTE(self, filename, sbin, mubin):
        norm_list_temp = open(filename, "r").readline().split()
        if len(norm_list_temp) == 5:
            self.DDnorm, self.DRnorm, self.RRnorm = [
                float(norm_list_temp[i]) for i in [1, 2, 4]
            ]
        elif len(norm_list_temp) == 4:
            self.DDnorm, self.DRnorm, self.RRnorm = [
                float(norm_list_temp[i]) for i in [1, 2, 3]
            ]
        else:
            raise ValueError(
                "norm_list_temp should be 3 or 4 elements long"
            )
        self.data = np.loadtxt(filename)
        if len(self.data) != sbin * mubin:
            raise ValueError("The length of data does not equal sbin*mubin")
        self.DD, self.DR, self.RR = [
            self.data[: sbin * mubin, row].reshape(sbin, mubin)
            for row in [3, 4, -1]
        ]
        self.Mu, self.S, self.xis = [
            self.data[: sbin * mubin, row].reshape(sbin, mubin)
            for row in [0, 1, 2]
        ]

    def _read_from_BINARY(self, filename, deal_with_0s0mu=True):
        source = joblib.load(filename)
        with_weight = source.get("with_weight", False)
        if with_weight:
            self.DD = source["DDwpairs"]
            self.DR = source["DRwpairs"]
            self.RR = source["RRwpairs"]
        else:
            self.DD = source["DDnpairs"]
            self.DR = source["DRnpairs"]
            self.RR = source["RRnpairs"]

        self.DDnorm = source["norm_d1d2"]
        self.DRnorm = source["norm_d1r2"]
        self.RRnorm = source["norm_r1r2"]

        self.xis = source.get("tpCF", None)
        if deal_with_0s0mu:
            self.DD[0,0] = 0.0 
            self.DR[0,0] = 0.0
            self.RR[0,0] = 1e-15
            if self.xis is not None:
                self.xis[0,0] = 0.0
        
        s_array = source.get("s_array", None)
        mu_array = source.get("mu_array", None)

        if s_array is None:
            sedges = source["sedges"]
            s_array = (sedges[1:] + sedges[:-1]) / 2.0
        if mu_array is None:
            muedges = source["muedges"]
            mu_array = (muedges[1:] + muedges[:-1]) / 2.0
        self.S, self.Mu = np.meshgrid(s_array, mu_array, indexing="ij")
    
    def integrate_tpcf(self, smin=6.0, smax=40.0, mumin=0.0, mumax=0.97, s_xis=False, intximu=True, mupack=1, is_norm=False, quick_return=False):
        """ A powerful function to integrate the tpcf

        Parameters
        smin : float
            The minimum scale
        smax : float
            The maximum scale
        mumin : float
            The minimum mu
        mumax : float
            The maximum mu
        s_xis : bool
            Whether to integrate over mu
        intximu : bool
            Whether to integrate over s
        is_norm: bool
            Whether to normalize
        re_calculate : bool
            Whether to re-calculate the tpcf based on DD, DR and RR
        """

        if hasattr(self, "s_array"):
            s_array = self.s_array
        else:
            s_array = np.mean(self.S, axis=1)
        
        if hasattr(self, "mu_array"):
            mu_array = self.mu_array
        else:
            mu_array = np.mean(self.Mu, axis=0)
        
        smin_index_source = np.where(s_array >= smin)[0]
        if len(smin_index_source) == 0:
            smin_index = 0
        else:
            smin_index = smin_index_source[0]
        smax_index_source = np.where(s_array >= smax)[0]
        if len(smax_index_source) == 0:
            smax_index = len(s_array)
        else:
            smax_index = smax_index_source[0]

        mumin_index_source = np.where(mu_array >= mumin)[0]
        if len(mumin_index_source) == 0:
            mumin_index = 0 
        else:
            mumin_index = mumin_index_source[0]
        mumax_index_source = np.where(mu_array >= mumax)[0]
        if len(mumax_index_source) == 0:
            mumax_index = len(mu_array) 
        else:
            mumax_index = mumax_index_source[0]
        
        DD_need = self.DD[smin_index: smax_index, mumin_index: mumax_index]
        DR_need = self.DR[smin_index: smax_index, mumin_index: mumax_index]
        RR_need = self.RR[smin_index: smax_index, mumin_index: mumax_index]
        if mupack > 1:
            DD_need = packarray2d(DD_need, mupack, axis=1)
            DR_need = packarray2d(DR_need, mupack, axis=1)
            RR_need = packarray2d(RR_need, mupack, axis=1)
            
        result_dict = {}
        if s_xis:
            s = s_array[smin_index: smax_index]
            xis_s = (np.sum(DD_need, axis=1) - 2 * np.sum(DR_need, axis=1) + np.sum(RR_need, axis=1)) / np.sum(RR_need, axis=1)
            if is_norm:
                xis_s = meannorm(xis_s)
            result_dict["s"] = s
            result_dict["xis_s"] = xis_s * s**2
            if not intximu and quick_return:
                return s, xis_s * s**2
        
        if intximu:
            mu = mu_array[mumin_index: mumax_index]
            s = s_array[smin_index: smax_index]
            if mupack > 1:
                mu = packarray1d(mu, mupack)
            Xis_need = (DD_need - 2 * DR_need + RR_need) / RR_need
            delta_s = np.mean(s[1:] - s[:-1])
            xis_mu = np.sum(Xis_need * delta_s, axis=0)
            if is_norm:
                xis_mu = meannorm(xis_mu)
            result_dict["mu"] = mu
            result_dict["xis_mu"] = xis_mu
            if not s_xis and quick_return:
                return mu, xis_mu

        return result_dict
    
    def get_specific_2d(self, smin=6.0, smax=40.0, mumin=0.0, mumax=0.97, norm_smax=45, mupack=1, is_norm=True):
        if hasattr(self, "s_array"):
            s_array = self.s_array
        else:
            s_array = np.mean(self.S, axis=1)
        
        if hasattr(self, "mu_array"):
            mu_array = self.mu_array
        else:
            mu_array = np.mean(self.Mu, axis=0)
        norm_smax_index_source = np.where(self.s_array >= norm_smax)[0]
        if len(norm_smax_index_source) == 0:
            norm_smax_index = len(s_array)
        else:
            norm_smax_index = norm_smax_index_source[0]
        
        S_mesh, Mu_mesh = np.meshgrid(s_array, mu_array, indexing="ij")
        if self.xis is None:
            self.xis = (self.DD - 2 * self.DR + self.RR) / self.RR
        if mupack > 1:
            S_mesh = packarray2d(S_mesh, mupack, axis=1)
            Mu_mesh = packarray2d(Mu_mesh, mupack, axis=1)
            self.xis = packarray2d(self.xis, mupack, axis=1)
            mu_array = packarray1d(mu_array, mupack)
        
        if is_norm:
            V_mesh = S_mesh**2 * self.xis
            norm_factor = 2 * np.pi * traz(V_mesh[:norm_smax_index, :], s_array[:norm_smax_index], mu_array)
        else:
            norm_factor = 1.0

        smin_index_source = np.where(s_array >= smin)[0]
        if len(smin_index_source) == 0:
            smin_index = 0
        else:
            smin_index = smin_index_source[0]
        smax_index_source = np.where(s_array >= smax)[0]
        if len(smax_index_source) == 0:
            smax_index = len(s_array)
        else:
            smax_index = smax_index_source[0]

        mumin_index_source = np.where(mu_array >= mumin)[0]
        if len(mumin_index_source) == 0:
            mumin_index = 0 
        else:
            mumin_index = mumin_index_source[0]
        mumax_index_source = np.where(mu_array >= mumax)[0]
        if len(mumax_index_source) == 0:
            mumax_index = len(mu_array) 
        else:
            mumax_index = mumax_index_source[0]

        result_dict = {
            "s": S_mesh[smin_index: smax_index, mumin_index: mumax_index],
            "mu": Mu_mesh[smin_index: smax_index, mumin_index: mumax_index],
            "xis": self.xis[smin_index: smax_index, mumin_index: mumax_index] / norm_factor,
        }
        return result_dict
        
    
    def cosmo_conv_simple(
        self, omstd=0.3071, wstd=-1, omwrong=0.5, wwrong=-1,
        redshift=0.0001,
        smin_mapping=3.0, smax_mapping=60.0,
    ):
        from ._AP_core import mapping_smudata_to_another_cosmology_simple
        from .base import Hz_jit, DA_jit
        if int(self.sbin) == 750 and int(self.mubin) == 600:
            raise ValueError(
                "Check your sbin and mubin of xismu. You are using simple convertor!"
            )
        redshift = max(redshift, 0.0001)
        
        Hstd = Hz_jit(redshift, omstd, wstd)
        Hnew = Hz_jit(redshift, omwrong, wwrong)
        DAstd = DA_jit(redshift, omstd, wstd)
        DAnew = DA_jit(redshift, omwrong, wwrong)

        data = np.concatenate(
            [
                self.DD[:, :, np.newaxis],
                self.DR[:, :, np.newaxis],
                self.RR[:, :, np.newaxis],
            ],
            axis=2,
        )
        new_data = mapping_smudata_to_another_cosmology_simple(
            data,
            DAstd,
            DAnew,
            Hstd,
            Hnew,
            deltamu=1.0 / self.mubin,
            max_mubin=self.mubin,
            smin_mapping=smin_mapping,
            smax_mapping=smax_mapping,
        )

        temp_DD = new_data[:, :, 0] * self.DDnorm
        temp_DR = new_data[:, :, 1] * self.DRnorm
        temp_RR = new_data[:, :, 2] * self.RRnorm

        return xismu(
            smax=self.smax,
            sbin=self.sbin,
            mubin=self.mubin,
            DDnorm=self.DDnorm,
            DRnorm=self.DRnorm,
            RRnorm=self.RRnorm,
            Mu=self.Mu,
            S=self.S,
            DD=temp_DD,
            DR=temp_DR,
            RR=temp_RR,
        )

    def cosmo_conv_DenseToSparse(
        self,
        omstd=0.3071,
        wstd=-1,
        omwrong=0.5,
        wwrong=-1,
        redshift=0.0001,
        sbin2=150,
        mubin2=120,
        smin_mapping=3.0,
        smax_mapping=60.0,
    ):
        from ._AP_core import mapping_smudata_to_another_cosmology_DenseToSparse, LoopStopException
        from .base import Hz_jit, DA_jit

        if int(self.sbin) == 150 and int(self.mubin) == 120:
            raise ValueError(
                "Check your sbin and mubin of xismu. You are using dense convertor!"
            )
        redshift = max(redshift, 0.0001)
        Hstd = Hz_jit(redshift, omstd, wstd)
        Hnew = Hz_jit(redshift, omwrong, wwrong)
        DAstd = DA_jit(redshift, omstd, wstd)
        DAnew = DA_jit(redshift, omwrong, wwrong)

        data = np.concatenate(
            [
                self.DD[:, :, np.newaxis],
                self.DR[:, :, np.newaxis],
                self.RR[:, :, np.newaxis],
            ],
            axis=2,
        )
        try:
            new_data = mapping_smudata_to_another_cosmology_DenseToSparse(
                data,
                DAstd,
                DAnew,
                Hstd,
                Hnew,
                deltas1=self.smax / self.sbin,
                deltamu1=1.0 / self.mubin,
                deltas2=self.smax / sbin2,
                deltamu2=1.0 / mubin2,
                smin_mapping=smin_mapping,
                smax_mapping=smax_mapping,
                compute_rows=[0, 1, 2],
            )
        except LoopStopException as e:
            raise LoopStopException(e)

        temp_DD = new_data[:, :, 0] * self.DDnorm
        temp_DR = new_data[:, :, 1] * self.DRnorm
        temp_RR = new_data[:, :, 2] * self.RRnorm

        return xismu(
            smax=self.smax,
            sbin=sbin2,
            mubin=mubin2,
            DDnorm=self.DDnorm,
            DRnorm=self.DRnorm,
            RRnorm=self.RRnorm,
            Mu=None, 
            S=None, 
            DD=temp_DD,
            DR=temp_DR,
            RR=temp_RR,
        )

def run_tpCF(data_catalog, random_catalog, sedges, mubin, with_weight, boxsize, run_parts=["all"], refine_factors=(2, 2, 1), output_dict=None, nthreads=1, verbose=False):
    """
    data_catalog & random_catalog: ndarray, three cols: [x, y, z](without weight) or four cols: [x, y, z, weight]
    run_parts: List. Support all, ALL, DD, DR, RR. If including all or ALL, run_parts will be set to ["DD", "DR", "RR"]
    """

    if "all" in run_parts or "ALL" in run_parts:
        run_parts = ["DD", "DR", "RR"]
    if isinstance(run_parts, str):
        run_parts = [run_parts]

    if "DD" in run_parts or "DR" in run_parts:
        if data_catalog.dtype != random_catalog.dtype:
            if data_catalog.dtype == np.float32:
                random_catalog = data_catalog.astype(np.float32)
            elif data_catalog.dtype == np.float64:
                random_catalog = data_catalog.astype(np.float64)
            else:
                raise ValueError("The data type of data_catalog and random_catalog should be np.float32 or np.float64")
        
    if output_dict is not None:
        output_DD = output_dict.get("DD", None)
        output_DR = output_dict.get("DR", None)
        output_RR = output_dict.get("RR", None)

    x_refine_factor, y_refine_factor, z_refine_factor = refine_factors
    result_dict = {
        "DD": None,
        "DR": None,
        "RR": None,
    }
    for run_part in run_parts:
        if run_part == "DD":
            if data_catalog is None:
                raise ValueError("DD: data_catalog is not set")
            else:
                autocorr = True 
                if verbose:
                    print("Now running DD")
                if with_weight:
                    DD_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], weights1=data_catalog[:,3], weight_type="pair_product", verbose=verbose, periodic=False, boxsize=boxsize, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    DD_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], verbose=verbose, periodic=False, boxsize=boxsize, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DD"] = DD_result
                if output_DD is not None:
                    joblib.dump(DD_result, output_DD)
            
        if run_part == "DR":
            if data_catalog is None or random_catalog is None:
                raise ValueError("DR: data_catalog or random_catalog is not set")
            else:
                if verbose:
                    print("Now running DR")
                autocorr = False 
                if with_weight:
                    DR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], weights1=data_catalog[:,3], X2=random_catalog[:,0], Y2=random_catalog[:,1], Z2=random_catalog[:,2], weights2=random_catalog[:,3], weight_type="pair_product", verbose = verbose, periodic=False, boxsize=boxsize, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    DR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], X2=random_catalog[:,0], Y2=random_catalog[:,1], Z2=random_catalog[:,2], verbose = verbose, periodic=False, boxsize=boxsize, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DR"] = DR_result
                if output_DR is not None:
                    joblib.dump(DR_result, output_DR)
                    
        if run_part == "RR":
            if random_catalog is None:
                raise ValueError("RR: random_catalog is not set")
            else:
                if verbose:
                    print("Now running RR")
                autocorr = True 
                if with_weight:
                    RR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=random_catalog[:,0], Y1=random_catalog[:,1], Z1=random_catalog[:,2], weights1=random_catalog[:,3], weight_type="pair_product", verbose = verbose, periodic=False, boxsize=boxsize, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    RR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=random_catalog[:,0], Y1=random_catalog[:,1], Z1=random_catalog[:,2], verbose = verbose, periodic=False, boxsize=boxsize, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["RR"] = RR_result
                if output_RR is not None:
                    joblib.dump(RR_result, output_RR)
    return result_dict

def run_tpCF_mock(mock_catalog, random_catalog, sedges, mubin, with_weight, run_parts=["all"], refine_factors=(2, 2, 1), output_dict=None, nthreads=1, verbose=False):
    """
    mock_catalog & random_catalog: ndarray, three cols: [x, y, z](without weight) or four cols: [x, y, z, weight]
    run_parts: List. Support all, ALL, DD, DR, RR. If including all or ALL, run_parts will be set to ["DD", "DR", "RR"]
    """

    if "all" in run_parts or "ALL" in run_parts:
        run_parts = ["DD", "DR", "RR"]
    if isinstance(run_parts, str):
        run_parts = [run_parts]

    if "DD" in run_parts or "DR" in run_parts:
        if mock_catalog.dtype != random_catalog.dtype:
            if mock_catalog.dtype == np.float32:
                random_catalog = mock_catalog.astype(np.float32, copy=False)
            elif mock_catalog.dtype == np.float64:
                random_catalog = mock_catalog.astype(np.float64, copy=False)
            else:
                raise ValueError("The data type of mock_catalog and random_catalog should be np.float32 or np.float64")
        
    if output_dict is not None:
        output_DD = output_dict.get("DD", None)
        output_DR = output_dict.get("DR", None)
        output_RR = output_dict.get("RR", None)

    x_refine_factor, y_refine_factor, z_refine_factor = refine_factors
    result_dict = {
        "DD": None,
        "DR": None,
        "RR": None,
    }
    for run_part in run_parts:
        if run_part == "DD":
            if mock_catalog is None:
                raise ValueError("DD: mock_catalog is not set")
            else:
                autocorr = True 
                if verbose:
                    print("Now running DD")
                if with_weight:
                    DD_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=mock_catalog[:,0], DEC1=mock_catalog[:,1], CZ1=mock_catalog[:,2], is_comoving_dist=True, weights1=mock_catalog[:,3], weight_type="pair_product", verbose=verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    DD_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=mock_catalog[:,0], DEC1=mock_catalog[:,1], CZ1=mock_catalog[:,2], is_comoving_dist=True, verbose=verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DD"] = DD_result
                if output_DD is not None:
                    joblib.dump(DD_result, output_DD)
            
        if run_part == "DR":
            if mock_catalog is None or random_catalog is None:
                raise ValueError("DR: mock_catalog or random_catalog is not set")
            else:
                if verbose:
                    print("Now running DR")
                autocorr = False 
                if with_weight:
                    DR_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=mock_catalog[:,0], DEC1=mock_catalog[:,1], CZ1=mock_catalog[:,2], is_comoving_dist=True, weights1=mock_catalog[:,3], RA2=random_catalog[:,0], DEC2=random_catalog[:,1], CZ2=random_catalog[:,2], weights2=random_catalog[:,3], weight_type="pair_product", verbose = verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    DR_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=mock_catalog[:,0], DEC1=mock_catalog[:,1], CZ1=mock_catalog[:,2], is_comoving_dist=True, RA2=random_catalog[:,0], DEC2=random_catalog[:,1], CZ2=random_catalog[:,2], verbose = verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DR"] = DR_result
                if output_DR is not None:
                    joblib.dump(DR_result, output_DR)
                    
        if run_part == "RR":
            if random_catalog is None:
                raise ValueError("RR: random_catalog is not set")
            else:
                if verbose:
                    print("Now running RR")
                autocorr = True 
                if with_weight:
                    RR_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=random_catalog[:,0], DEC1=random_catalog[:,1], CZ1=random_catalog[:,2], is_comoving_dist=True, weights1=random_catalog[:,3], weight_type="pair_product", verbose = verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    RR_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=random_catalog[:,0], DEC1=random_catalog[:,1], CZ1=random_catalog[:,2], is_comoving_dist=True, verbose = verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["RR"] = RR_result
                if output_RR is not None:
                    joblib.dump(RR_result, output_RR)
    return result_dict

def cal_tpCF_from_pairs(DD_result, DR_result, RR_result, data, random, sbin, mubin, with_weight=False):
    result_dict = {}
    result_dict["DDnpairs"] = DD_result["npairs"].reshape(sbin, mubin)
    result_dict["DRnpairs"] = DR_result["npairs"].reshape(sbin, mubin)
    result_dict["RRnpairs"] = RR_result["npairs"].reshape(sbin, mubin)
    if with_weight:
        result_dict["DDwpairs"] = (DD_result["npairs"] * DD_result["weightavg"]).reshape(sbin, mubin)
        result_dict["DRwpairs"] = (DR_result["npairs"] * DR_result["weightavg"]).reshape(sbin, mubin)
        result_dict["RRwpairs"] = (RR_result["npairs"] * RR_result["weightavg"]).reshape(sbin, mubin)
        result_dict["with_weight"] = True 
    else:
        result_dict["with_weight"] = False

    if with_weight:
        DD = result_dict["DDwpairs"]
        DR = result_dict["DRwpairs"]
        RR = result_dict["RRwpairs"]
    else:
        DD = result_dict["DDnpairs"]
        DR = result_dict["DRnpairs"]
        RR = result_dict["RRnpairs"]

    result_dict["sedges"] = np.append(DD_result["smin"].reshape(sbin, mubin)[:,0], DD_result["smax"].reshape(sbin, mubin)[-1,0])
    mumax_str = "mumax" if "mumax" in DD_result.dtype.names else "mu_max"
    result_dict["muedges"] = np.append([0], DD_result[mumax_str].reshape(sbin, mubin)[0])

    dataNum = data.shape[0]
    randomNum = random.shape[0]
    if with_weight:
        randomWeight = random[:,3]
        dataWeight = data[:,3]

    sum_wd = np.sum(dataWeight) if with_weight else dataNum
    sum_wr = np.sum(randomWeight) if with_weight else randomNum
    sum_wd2 = np.sum(dataWeight**2) if with_weight else dataNum
    sum_wr2 = np.sum(randomWeight**2) if with_weight else randomNum
    sum_wr_2 = sum_wr 
    sum_wd_2 = sum_wd 

    norm_d1d2 = sum_wd * sum_wd - sum_wd2 
    norm_r1r2 = sum_wr * sum_wr - sum_wr2
    norm_d1r2 = sum_wd * sum_wr_2 
    norm_r1d2 = sum_wr * sum_wd_2

    if abs(result_dict["sedges"][0] - 0.0) < 1e-10 and abs(result_dict["muedges"][0] - 0.0) < 1e-10:
        DD[0,0] -= sum_wd 
        RR[0,0] -= sum_wr 

    dd1d2 = DD / norm_d1d2
    dd1r2 = DR / norm_d1r2
    dr1d2 = DR / norm_r1d2
    dr1r2 = RR / norm_r1r2

    dr1r2_remove_0 = np.copy(dr1r2)
    dr1r2_remove_0[dr1r2_remove_0 == 0] = 1e-15

    result_dict.update({
        "norm_d1d2": norm_d1d2,
        "norm_r1r2": norm_r1r2,
        "norm_d1r2": norm_d1r2,
        "norm_r1d2": norm_r1d2,
        "tpCF": (dd1d2 - dr1d2 - dd1r2 + dr1r2) / dr1r2_remove_0,
    })

    return result_dict