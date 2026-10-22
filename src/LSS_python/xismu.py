"""Correlation-function result objects and their analysis tools.

This module owns xismu serialization, integrations, rebinning, averaging and
profile differences. It does not depend on the counting or covariance modules.
AP geometry and optional compressors are imported only by the methods that
use them. Corrfunc measurement entry points remain in LSS_python.tpcf.
"""

import os
import stat
import tempfile
import warnings
from contextvars import ContextVar

import joblib
import numpy as np

from .base import traz

__all__ = [
    "xismu", "meannorm", "packarray1d", "packarray2d", "cal_convert_factor",
    "average_xismus_dict_list", "get_diff_array", "get_diff_array_test",
    "get_diff_array_2d", "create_xismu_from_pairs", "migrate_xismu_file",
]

def meannorm(X, axis=0):
    if len(X.shape) == 1:
        axis = None
    mean_value = np.mean(X, axis=axis)
    return 1 + (X - mean_value) / np.abs(mean_value)

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

def cal_convert_factor(n, rat):
    """
    Compute the conversion matrix that transforms a full covariance matrix
    to the covariance matrix after applying packarray1d with given ratio.

    The packing operation is a linear transformation: X_packed = P @ X,
    where P is an averaging matrix of shape (floor(n/rat), n).

    For covariance matrices: Cov_packed = P @ Cov_full @ P.T

    Parameters
    ----------
    n : int
        Original length of the data array.
    rat : int
        Packing ratio (number of elements to average together).

    Returns
    -------
    convert_matrix : ndarray
        The conversion matrix P of shape (floor(n/rat), n).
        Can be used as: Cov_packed = P @ Cov_full @ P.T
    """
    m = int(n / rat)  # packed length, floor division (same as packarray2d)
    convert_matrix = np.zeros((m, n))

    for i in range(m):
        # Each row averages rat consecutive elements
        convert_matrix[i, i * rat : (i + 1) * rat] = 1.0 / rat

    return convert_matrix

class xismu(object):
    def __init__(
        self,
        smax=150, sbin=150, mubin=120,
        DDnorm=None, DRnorm=None, RRnorm=None,
        S=None, Mu=None,
        DD=None, DR=None, RR=None,
        set_data=True,
        sum_wd=None, sum_wr=None, sum_wd2=None, sum_wr2=None,
    ):
        """
        data_type: CUTE, BINARY(As Corrfunc)
        Note: only support DR instead of D1R2 and D2R1 now.
        filename: filename of data. If None, DD, DR, RR before normalization and other parameters are required.
        """
        self.smax, self.sbin, self.mubin = smax, sbin, mubin
        self.sum_wd, self.sum_wr = sum_wd, sum_wr
        self.sum_wd2, self.sum_wr2 = sum_wd2, sum_wr2

        if set_data:
            self.DDnorm, self.DRnorm, self.RRnorm = DDnorm, DRnorm, RRnorm
            self.S = S
            self.Mu = Mu
            self.DD = DD.reshape((self.sbin, self.mubin)) if DD is not None else None
            self.DR = DR.reshape((self.sbin, self.mubin)) if DR is not None else None
            self.RR = RR.reshape((self.sbin, self.mubin)) if RR is not None else None
            self.DD = self.DD / self.DDnorm if self.DD is not None else None
            self.DR = self.DR / self.DRnorm if self.DR is not None else None
            self.RR = self.RR / self.RRnorm if self.RR is not None else None
            if self.RR is not None:
                self.RR[self.RR == 0] = 1e-15
        self.xis = None
        self.s_array = None
        self.mu_array = None

        self.filename = None
        self.data_type = None

    def _check_data_not_None(self, types=["s_mu_array", "pairs"]):
        if "s_mu_array" in types:
            if self.s_array is None or self.mu_array is None:
                raise ValueError("s_array and mu_array must be set")
        if "pairs" in types:
            if self.DD is None or self.DR is None or self.RR is None:
                raise ValueError("DD, DR, RR must be set")
            if self.DDnorm is None or self.DRnorm is None or self.RRnorm is None:
                raise ValueError("DDnorm, DRnorm, RRnorm must be set")

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
            if self.DD is None:
                raise ValueError("DD is None")
            if self.DD.shape[0] != self.sbin or self.DD.shape[1] != self.mubin:
                raise ValueError("sbin or mubin is not equal to the data")
        else:
            raise ValueError("data_type must be CUTE or BINARY")

        if self.DD is None or self.DR is None or self.RR is None:
            raise ValueError("DD, DR, RR must be set")
        if self.DDnorm is None or self.DRnorm is None or self.RRnorm is None:
            raise ValueError("DDnorm, DRnorm, RRnorm must be set")

        self._check_data_not_None(types=["pairs", ])
        self.DD = self.DD / self.DDnorm
        self.DR = self.DR / self.DRnorm
        self.RR = self.RR / self.RRnorm
        if self.RR is not None:
            self.RR[self.RR == 0.0] = 1e-15

        if self.S is None or self.Mu is None:
            raise ValueError("S and Mu must be included in the data when using load")
        return self

    def save(self, filename, with_weight=True):
        result_dict = {}
        result_dict["with_weight"] = with_weight
        self._check_data_not_None(types=["pairs", ])
        if self.DD is None or self.DR is None or self.RR is None:
            raise ValueError("DD, DR, RR must be set")
        if self.DDnorm is None or self.DRnorm is None or self.RRnorm is None:
            raise ValueError("DDnorm, DRnorm, RRnorm must be set")
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
        for name in ("sum_wd", "sum_wr", "sum_wd2", "sum_wr2"):
            result_dict[name] = getattr(self, name, None)

        result_dict["s_array"] = self.S[:,0] if self.S is not None else self.s_array
        result_dict["mu_array"] = self.Mu[0] if self.Mu is not None else self.mu_array
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
        for name in ("sum_wd", "sum_wr", "sum_wd2", "sum_wr2"):
            setattr(self, name, source.get(name))

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

    def get_xis(self):
        if self.DD is None or self.DR is None or self.RR is None:
            raise ValueError("DD, DR, RR must be set before getting xis")
        if self.xis is None:
            self.xis = (self.DD - 2.0 * self.DR + self.RR) / self.RR
        return self.xis

    def intximu(self, smin=6.0, smax=40.0, mumin=0.0, mumax=0.97,
                with_s2=False, mupack=1, is_norm=False,
                remove_last_one=True) -> tuple:
        """Integrate over ``s`` after evaluating the correlation function.

        This method preserves the original integration order: evaluate xi on
        the original mu bins, integrate over s, normalize, and pack the
        resulting mu-dependent curve last.
        """
        if hasattr(self, "s_array"):
            s_array = self.s_array
            if s_array is None:
                if self.S is None:
                    raise ValueError("S must be set before getting s_array")
                s_array = np.mean(self.S, axis=1)
        else:
            if self.S is None:
                raise ValueError("S must be set before getting s_array")
            s_array = np.mean(self.S, axis=1)

        if hasattr(self, "mu_array"):
            mu_array = self.mu_array
            if mu_array is None:
                if self.Mu is None:
                    raise ValueError("Mu must be set before getting mu_array")
                mu_array = np.mean(self.Mu, axis=0)
        else:
            if self.Mu is None:
                raise ValueError("Mu must be set before getting mu_array")
            mu_array = np.mean(self.Mu, axis=0)

        if s_array is None or mu_array is None:
            raise ValueError("s_array and mu_array must be set before integrating")
        if self.DD is None or self.DR is None or self.RR is None:
            raise ValueError("DD, DR, RR must be set before integrating")

        smin_index_source = np.where(s_array >= smin)[0]
        smin_index = smin_index_source[0] if len(smin_index_source) else 0
        smax_index_source = np.where(s_array >= smax)[0]
        smax_index = smax_index_source[0] if len(smax_index_source) else len(s_array)

        mumin_index_source = np.where(mu_array >= mumin)[0]
        mumin_index = mumin_index_source[0] if len(mumin_index_source) else 0
        mumax_index_source = np.where(mu_array >= mumax)[0]
        mumax_index = mumax_index_source[0] if len(mumax_index_source) else len(mu_array)

        s = s_array[smin_index:smax_index]
        mu = mu_array[mumin_index:mumax_index]
        DD_need = self.DD[smin_index:smax_index, mumin_index:mumax_index]
        DR_need = self.DR[smin_index:smax_index, mumin_index:mumax_index]
        RR_need = self.RR[smin_index:smax_index, mumin_index:mumax_index]

        xis_need = (DD_need - 2.0 * DR_need + RR_need) / RR_need
        if with_s2:
            xis_need = xis_need * s[:, np.newaxis] ** 2
        xis_mu = np.mean(xis_need, axis=0)

        if is_norm:
            xis_mu = meannorm(xis_mu)
            if remove_last_one:
                xis_mu = xis_mu[:-1]
                mu = mu[:-1]

        if mupack > 1:
            xis_mu = packarray1d(xis_mu, mupack)
            mu = packarray1d(mu, mupack)

        return mu, xis_mu

    def integrate_tpcf(self, smin=6.0, smax=40.0, mumin=0.0, mumax=0.97, s_xis=False, intximu=False, with_s2=False, mupack=1, is_norm=False, quick_return=True, remove_last_one=True) -> dict | tuple:
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
        with_s2 : bool
            Only be valid when intximu is True. Use s^2 xi(s, mu) to integrate
        is_norm: bool
            Whether to normalize
        remove_last_one: bool
            Only be valid when intximu=True and is_norm=True
        mupack: int
            The number of mu bins to pack together for integration
        """

        if not s_xis and not intximu:
            raise ValueError("s_xis or intximu must be True")

        if hasattr(self, "s_array"):
            s_array = self.s_array
            if s_array is None:
                if self.S is None:
                    raise ValueError("S must be set before getting s_array")
                s_array = np.mean(self.S, axis=1)
        else:
            if self.S is None:
                raise ValueError("S must be set before getting s_array")
            s_array = np.mean(self.S, axis=1)

        if hasattr(self, "mu_array"):
            mu_array = self.mu_array
            if mu_array is None:
                if self.Mu is None:
                    raise ValueError("Mu must be set before getting mu_array")
                mu_array = np.mean(self.Mu, axis=0)
        else:
            if self.Mu is None:
                raise ValueError("Mu must be set before getting mu_array")
            mu_array = np.mean(self.Mu, axis=0)

        if s_array is None or mu_array is None:
            raise ValueError("s_array and mu_array must be set before getting xis")

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

        if self.DD is None or self.DR is None or self.RR is None:
            raise ValueError("DD, DR, RR must be set before getting xis")

        DD_need = self.DD[smin_index: smax_index, mumin_index: mumax_index]
        DR_need = self.DR[smin_index: smax_index, mumin_index: mumax_index]
        RR_need = self.RR[smin_index: smax_index, mumin_index: mumax_index]

        result_dict = {}
        if s_xis:
            s = s_array[smin_index: smax_index]
            xis_s = (np.sum(DD_need, axis=1) - 2 * np.sum(DR_need, axis=1) + np.sum(RR_need, axis=1)) / np.sum(RR_need, axis=1)
            if is_norm:
                xis_s = meannorm(xis_s)
            if mupack > 1:
                s = packarray1d(s, mupack)
                xis_s = packarray1d(xis_s, mupack)
            result_dict["s"] = s
            result_dict["xis_s"] = xis_s * s**2
            if not intximu and quick_return:
                return s, xis_s * s**2

        if intximu:
            # Pack pair counts first, then evaluate xi and integrate over s.
            # Normalization is deliberately applied after the integration.
            mu = mu_array[mumin_index: mumax_index]
            s = s_array[smin_index: smax_index]
            if mupack > 1:
                DD_need = packarray2d(DD_need, mupack, axis=1)
                DR_need = packarray2d(DR_need, mupack, axis=1)
                RR_need = packarray2d(RR_need, mupack, axis=1)
                mu = packarray1d(mu, mupack)
            Xis_need = (DD_need - 2 * DR_need + RR_need) / RR_need
            if with_s2:
                Xis_need = Xis_need * s[:, np.newaxis]**2
            xis_mu = np.mean(Xis_need, axis=0)
            if is_norm:
                xis_mu = meannorm(xis_mu)
                if remove_last_one:
                    xis_mu = xis_mu[:-1]
                    mu = mu[:-1]
            result_dict["mu"] = mu
            result_dict["xis_mu"] = xis_mu
            if not s_xis and quick_return:
                return mu, xis_mu

        return result_dict

    def get_specific_2d(self, smin=6.0, smax=40.0, mumin=0.0, mumax=0.97, norm_smax=45, is_norm=True):
        if hasattr(self, "s_array"):
            s_array = self.s_array
        else:
            if self.S is None:
                raise ValueError("S must be set before getting s_array")
            s_array = np.mean(self.S, axis=1)

        if hasattr(self, "mu_array"):
            mu_array = self.mu_array
        else:
            if self.Mu is None:
                raise ValueError("Mu must be set before getting mu_array")
            mu_array = np.mean(self.Mu, axis=0)

        if s_array is None or mu_array is None:
            raise ValueError("s_array and mu_array must be set")
        norm_smax_index_source = np.where(s_array >= norm_smax)[0]
        if len(norm_smax_index_source) == 0:
            norm_smax_index = len(s_array)
        else:
            norm_smax_index = norm_smax_index_source[0]

        S_mesh, Mu_mesh = np.meshgrid(s_array, mu_array, indexing="ij")
        if not hasattr(self, "xis") or self.xis is None:
            if self.DD is None or self.DR is None or self.RR is None:
                raise ValueError("DD, DR, RR must be set before getting xis")
            self.xis = (self.DD - 2 * self.DR + self.RR) / self.RR

        if is_norm:
            V_mesh = S_mesh**2 * self.xis
            norm_factor = 2.0 * np.pi * traz(V_mesh[:norm_smax_index, :], s_array[:norm_smax_index], mu_array)
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
        smin_mapping=3.0, smax_mapping=60.0, wastd=0.0, wawrong=0.0,
    ):
        """Interpolate counts after normalizing redshift and optional wa inputs."""
        from ._AP_core import mapping_smudata_to_another_cosmology_simple
        from ._ap_dense import normalize_background_inputs
        from .base import Hz_w0wa_jit, DA_jit
        if int(self.sbin) == 750 and int(self.mubin) == 600:
            raise ValueError(
                "Check your sbin and mubin of xismu. You are using simple convertor!"
            )
        redshift, wastd, wawrong = normalize_background_inputs(
            redshift, omstd, wstd, omwrong, wwrong, wastd, wawrong)

        Hstd = Hz_w0wa_jit(redshift, omstd, wstd, wastd)
        Hnew = Hz_w0wa_jit(redshift, omwrong, wwrong, wawrong)
        DAstd = DA_jit(redshift, omstd, wstd, wastd)
        DAnew = DA_jit(redshift, omwrong, wwrong, wawrong)

        if self.DD is None or self.DR is None or self.RR is None:
            raise ValueError("DD, DR, RR must be set before converting")

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
            sum_wd=getattr(self, "sum_wd", None),
            sum_wr=getattr(self, "sum_wr", None),
            sum_wd2=getattr(self, "sum_wd2", None),
            sum_wr2=getattr(self, "sum_wr2", None),
            Mu=self.Mu,
            S=self.S,
            DD=temp_DD,
            DR=temp_DR,
            RR=temp_RR,
        )

    def cosmo_conv_DenseToSparse(
        self, omstd=0.3071, wstd=-1, omwrong=0.5, wwrong=-1,
        redshift=0.0001, sbin2=150, mubin2=120,
        smin_mapping=3.0, smax_mapping=60.0, assistant_xismu=None,
        wastd=0.0, wawrong=0.0, c_api=True, *, convert_algorithm="polygon",
    ):
        """Rebin normalized counts using uniform mapped (s, mu) areas.

        Existing output-size arguments remain available without an assistant.
        With an assistant, its coordinates must match the requested output size.
        ``c_api=False`` uses the equivalent Numba geometry.
        """
        from ._ap_dense import ap_geometry, convert_xismu, validate_algorithm

        validate_algorithm(convert_algorithm)
        geometry = ap_geometry(redshift, omstd, wstd, omwrong, wwrong, wastd, wawrong)
        return convert_xismu(self, geometry, sbin2, mubin2, smin_mapping, smax_mapping,
                             assistant_xismu, c_api, convert_algorithm)

def average_xismus_dict_list(xismus_dict_list, debug=False):
    """Average pair counts in a list of xismu dictionaries.

    The stored DD, DR and RR arrays are already normalized. Their original
    pair counts are recovered before averaging, and the output normalizations
    are recalculated from averaged weight sums when available. Otherwise, the
    existing normalization constants are averaged. When ``debug`` is True, a
    warning is emitted in that fallback case.
    """
    if not xismus_dict_list:
        raise ValueError("xismus_dict_list must not be empty")

    keys = xismus_dict_list[0].keys()
    if any(xismus_dict.keys() != keys for xismus_dict in xismus_dict_list[1:]):
        raise ValueError("all dictionaries must have the same keys")

    result = {}
    count = len(xismus_dict_list)
    moment_names = ("sum_wd", "sum_wr", "sum_wd2", "sum_wr2")
    for key in keys:
        reference = xismus_dict_list[0][key]
        if not isinstance(reference, xismu):
            raise TypeError(f"value for {key!r} must be an xismu")
        shape = (reference.sbin, reference.mubin)
        pairs = {name: np.zeros(shape, dtype=float) for name in ("DD", "DR", "RR")}
        norm_sums = {name: 0.0 for name in pairs}
        moment_sums = {name: 0.0 for name in moment_names}
        has_all_moments = True

        for realization_dict in xismus_dict_list:
            realization = realization_dict[key]
            if not isinstance(realization, xismu):
                raise TypeError(f"value for {key!r} must be an xismu")
            if (realization.sbin, realization.mubin) != shape or not np.isclose(
                realization.smax, reference.smax
            ):
                raise ValueError(f"incompatible bins for {key!r}")
            for coordinate in ("S", "Mu"):
                first_grid = getattr(reference, coordinate)
                grid = getattr(realization, coordinate)
                if (first_grid is None) != (grid is None) or (
                    first_grid is not None and not np.array_equal(grid, first_grid)
                ):
                    raise ValueError(f"incompatible {coordinate} grids for {key!r}")

            norms = (realization.DDnorm, realization.DRnorm, realization.RRnorm)
            if any(norm is None or not np.isfinite(norm) or norm <= 0 for norm in norms):
                raise ValueError(f"missing or invalid pair normalization for {key!r}")

            moments = tuple(getattr(realization, name, None) for name in moment_names)
            if any(value is None for value in moments):
                has_all_moments = False
            else:
                if any(not np.isfinite(value) for value in moments):
                    raise ValueError(f"invalid weight sums for {key!r}")
                for name, value in zip(moment_names, moments):
                    moment_sums[name] += value

            for name, norm in zip(pairs, norms):
                norm_sums[name] += norm
                values = getattr(realization, name)
                if values is None or values.shape != shape:
                    raise ValueError(f"missing or incompatible {name} for {key!r}")
                if name == "RR":
                    values = np.where(values == 1e-15, 0.0, values)
                pairs[name] += values * norm

        if has_all_moments:
            moments = {name: total / count for name, total in moment_sums.items()}
            wd, wr = moments["sum_wd"], moments["sum_wr"]
            norms = {
                "DD": wd * wd - moments["sum_wd2"],
                "DR": wd * wr,
                "RR": wr * wr - moments["sum_wr2"],
            }
        else:
            if debug:
                warnings.warn(
                    f"Missing weight sums for {key!r}; averaging existing normalization constants",
                    RuntimeWarning,
                    stacklevel=2,
                )
            moments = {name: None for name in moment_names}
            norms = {name: total / count for name, total in norm_sums.items()}

        if any(not np.isfinite(norm) or norm <= 0 for norm in norms.values()):
            raise ValueError(f"averaged pair normalization is invalid for {key!r}")

        result[key] = xismu(
            smax=reference.smax,
            sbin=reference.sbin,
            mubin=reference.mubin,
            DDnorm=norms["DD"],
            DRnorm=norms["DR"],
            RRnorm=norms["RR"],
            S=reference.S,
            Mu=reference.Mu,
            DD=pairs["DD"] / count,
            DR=pairs["DR"] / count,
            RR=pairs["RR"] / count,
            **moments,
        )

    return result

def get_diff_array(tpcf_dict_list, snap_ids, shift=0, return_mu=False,
                   compressor=None, intximu_new=False, mupack=1, remove_last_one=True,
                   *, is_norm=True, norm_method="regu", eta=0.2, **kwargs) -> np.ndarray:
    """Return the difference of two angular profiles.

    The built-in integration paths always return raw profiles. When
    ``is_norm`` is true, each complete profile is normalized and optionally
    shortened before the two snapshots are subtracted. ``norm_method=None``
    reproduces the original mean normalization, while ``"regu"`` and
    ``"regu_hard"`` use smooth and hard regularization, respectively.

    intximu_new: bool, default False
        Use the standalone :meth:`xismu.intximu` implementation, which keeps
        the original integrate-tpcf ordering. This cannot be used with
        ``integrate_tpcf_func``.

    is_norm: bool, default True
        Disable all built-in normalization when false.
    norm_method: None or {"regu", "regu_hard"}, default "regu"
        ``None`` preserves the original ``meannorm`` calculation.
    eta: float, default 0.2
        Coefficient of the full-profile standard deviation in regularized
        normalization.

    kwargs:
        smin, smax: float, default 6.0, 40.0
        mumax: float, default 0.97
        integrate_tpcf_func: optional custom integrator. Its output is used
            unchanged; it owns its normalization and bin removal.
        integrate_tpcf_kwargs: dict, default None and use {}
    """
    if "remove_after_diff" in kwargs:
        raise TypeError("remove_after_diff has been removed and is no longer supported")
    if norm_method is not None and (
        not isinstance(norm_method, str) or norm_method not in ("regu", "regu_hard")
    ):
        raise ValueError("norm_method must be None, 'regu', or 'regu_hard'")
    if (
        not np.isscalar(eta) or not np.isrealobj(eta)
        or not isinstance(eta, (int, float, np.number))
        or not np.isfinite(eta) or eta < 0
    ):
        raise ValueError("eta must be a finite non-negative scalar")

    integrate_tpcf_func = kwargs.get("integrate_tpcf_func", None)
    if intximu_new and integrate_tpcf_func is not None:
        raise ValueError(
            "intximu_new cannot be used with integrate_tpcf_func; "
            "the custom integration function must select its own behavior"
        )

    if integrate_tpcf_func is None:
        smin = kwargs.get("smin", 6.0)
        smax = kwargs.get("smax", 40.0)
        mumax = kwargs.get("mumax", 0.97)
    else:
        integrate_tpcf_kwargs = kwargs.get("integrate_tpcf_kwargs", {})

    def normalize_profile(mu, values):
        mu = np.asarray(mu)
        values = np.asarray(values, dtype=float)
        if norm_method is None:
            normalized = meannorm(values)
        else:
            if values.ndim != 1 or mu.shape != values.shape or not np.all(np.isfinite(values)):
                raise ValueError("integrated profile must contain finite one-dimensional values")
            if values.size == 0:
                raise ValueError("integrated profile has no angular bins")
            mean = float(np.mean(values))
            spread = float(np.std(values))
            scale = (
                np.hypot(mean, eta * spread)
                if norm_method == "regu" else max(abs(mean), eta * spread)
            )
            if scale == 0.0:
                raise ValueError("normalization scale is zero; the profile has no defined shape")
            normalized = 1.0 + (values - mean) / scale
        if remove_last_one:
            return mu[:-1], normalized[:-1]
        return mu, normalized

    def integrate_profile(obj):
        if intximu_new:
            # intximu historically normalizes before angular packing.
            raw_mupack = 1 if is_norm else mupack
            mu, values = obj.intximu(
                smin=smin, smax=smax, mupack=raw_mupack, is_norm=False,
                mumax=mumax, remove_last_one=False,
            )
            if is_norm:
                mu, values = normalize_profile(mu, values)
                if mupack > 1:
                    mu = packarray1d(mu, mupack)
                    values = packarray1d(values, mupack)
            return mu, values

        mu, values = obj.integrate_tpcf(
            smin=smin, smax=smax, intximu=True, mupack=mupack,
            is_norm=False, mumax=mumax, remove_last_one=False,
        )
        if is_norm:
            return normalize_profile(mu, values)
        return mu, values

    if isinstance(tpcf_dict_list, dict):
        tpcf_dict_list = [tpcf_dict_list, ]
    snap1, snap2 = snap_ids[0], snap_ids[1]

    # Ensure tpcf_dict_list[0][snap1] is a list, tuple, or array
    first_snap_data = tpcf_dict_list[0][snap1]
    if not isinstance(first_snap_data, (list, tuple, np.ndarray)):
        first_snap_data = [first_snap_data]
        single_xismu = True
    else:
        single_xismu = False

    size = len(first_snap_data)
    if size == 1:
        shift = 0
    tpcf_diff_list_list = []

    if compressor is None:
        use_compressor = False
    else:
        use_compressor = True

    for i_list in range(len(tpcf_dict_list)):
        tpcf_diff_list = []
        for i in range(size):
            i_shift = i + shift
            if i_shift >= size:
                i_shift -= size
            xismu_first = tpcf_dict_list[i_list][snap1][i] if not single_xismu else tpcf_dict_list[i_list][snap1]
            xismu_second = tpcf_dict_list[i_list][snap2][i_shift] if not single_xismu else tpcf_dict_list[i_list][snap2]
            if integrate_tpcf_func is not None:
                mu_temp_1, xi_mu_temp_1 = integrate_tpcf_func(xismu_first, **integrate_tpcf_kwargs)
                mu_temp_2, xi_mu_temp_2 = integrate_tpcf_func(xismu_second, **integrate_tpcf_kwargs)
            else:
                mu_temp_1, xi_mu_temp_1 = integrate_profile(xismu_first)
                mu_temp_2, xi_mu_temp_2 = integrate_profile(xismu_second)

            xi_mu_temp_diff = xi_mu_temp_1 - xi_mu_temp_2
            tpcf_diff_list.append(xi_mu_temp_diff)
        tpcf_diff_list_list.append(np.array(tpcf_diff_list))

    diff_array = np.concatenate(tpcf_diff_list_list, axis=1)

    # Apply PCA transformation if needed
    if use_compressor:
        from .compressor import Compressor, CompressorSeries
        if isinstance(compressor, CompressorSeries) or isinstance(compressor, Compressor):
            result_array = compressor.transform(diff_array)
        else:
            raise ValueError("compressors_list must be a list of Compressor or a single Compressor")
    else:
        result_array = diff_array

    if result_array.shape[0] == 1:
        result_array = result_array[0]

    if return_mu:
        return np.vstack((mu_temp_1, result_array))
    else:
        return result_array

def get_diff_array_test(tpcf_dict_list, snap_ids, shift=0, return_mu=False,
                        compressor=None, intximu_new=False, mupack=1,
                        remove_last_one=True, eta=0.2,
                        use_hard_floor=False, **kwargs) -> np.ndarray:
    """Subtract two snapshots after regularized mean-one normalization.

    For each full angular profile ``x`` with mean ``m`` and standard deviation
    ``s``, return ``1 + (x - m) / scale``. The default smooth scale is
    ``hypot(m, eta * s)``; ``use_hard_floor=True`` selects
    ``max(abs(m), eta * s)`` instead. Thus ``eta`` is the coefficient of the
    standard deviation in both methods. Each full profile has mean one before
    the optional last-bin removal.

    Integration and pairing follow :func:`get_diff_array`. Normalize the full
    profile before optionally removing its last bin and subtracting snapshots.
    """
    if "is_norm" in kwargs:
        raise TypeError("get_diff_array_test always uses regularized normalization")
    if "remove_after_diff" in kwargs:
        raise TypeError("remove_after_diff has been removed and is no longer supported")
    if not np.isscalar(eta) or not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be a finite non-negative scalar")
    if not isinstance(use_hard_floor, (bool, np.bool_)):
        raise TypeError("use_hard_floor must be a boolean")

    integrate_tpcf_func = kwargs.get("integrate_tpcf_func")
    integrate_tpcf_kwargs = kwargs.get("integrate_tpcf_kwargs", {})
    if intximu_new and integrate_tpcf_func is not None:
        raise ValueError(
            "intximu_new cannot be used with integrate_tpcf_func; "
            "the custom integration function must select its own behavior"
        )
    if integrate_tpcf_func is None and not intximu_new:
        return get_diff_array(
            tpcf_dict_list, snap_ids, shift=shift, return_mu=return_mu,
            compressor=compressor, mupack=mupack, remove_last_one=remove_last_one,
            norm_method="regu_hard" if use_hard_floor else "regu", eta=eta,
            **kwargs,
        )
    smin = kwargs.get("smin", 6.0)
    smax = kwargs.get("smax", 40.0)
    mumax = kwargs.get("mumax", 0.97)

    def integrate_and_normalize(obj):
        if integrate_tpcf_func is not None:
            mu, values = integrate_tpcf_func(obj, **integrate_tpcf_kwargs)
        elif intximu_new:
            mu, values = obj.intximu(
                smin=smin, smax=smax, mupack=mupack, is_norm=False,
                mumax=mumax, remove_last_one=False,
            )
        else:
            mu, values = obj.integrate_tpcf(
                smin=smin, smax=smax, intximu=True, mupack=mupack,
                is_norm=False, mumax=mumax, remove_last_one=False,
            )
        mu = np.asarray(mu)
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or mu.shape != values.shape or not np.all(np.isfinite(values)):
            raise ValueError("integrated profile must contain finite one-dimensional values")
        if values.size == 0:
            raise ValueError("integrated profile has no angular bins")
        mean = float(np.mean(values))
        spread = float(np.std(values))
        scale = (
            max(abs(mean), eta * spread)
            if use_hard_floor else np.hypot(mean, eta * spread)
        )
        if scale == 0.0:
            raise ValueError("normalization scale is zero; the profile has no defined shape")
        normalized = 1.0 + (values - mean) / scale
        if remove_last_one:
            return mu[:-1], normalized[:-1]
        return mu, normalized

    return get_diff_array(
        tpcf_dict_list, snap_ids, shift=shift, return_mu=return_mu,
        compressor=compressor, mupack=mupack, remove_last_one=remove_last_one,
        integrate_tpcf_func=integrate_and_normalize,
    )

def get_diff_array_2d(tpcf_dict_list, snap_ids, shift=5, compressor=None, **kwargs) -> np.ndarray:
    """
    Get difference array from 2D tpcf data using get_specific_2d method.

    Parameters
    tpcf_dict_list : dict or list of dict
        Dictionary or list of dictionaries containing tpcf data
    snap_ids : list
        List of two snapshot IDs to compare
    shift : int, default 5
        Shift index for comparison
    compressor : Compressor or CompressorSeries, optional
        Compressor for PCA transformation
    **kwargs : dict
        Additional arguments for get_specific_2d:
        - smin, smax: float, default 6.0, 40.0
        - mumin, mumax: float, default 0.0, 0.97
        - norm_smax: float, default 45.0

    Returns
    np.ndarray
        Difference array after processing and optional compression
    """
    # Get parameters for get_specific_2d
    smin = kwargs.get("smin", 6.0)
    smax = kwargs.get("smax", 40.0)
    mumin = kwargs.get("mumin", 0.0)
    mumax = kwargs.get("mumax", 0.97)
    norm_smax = kwargs.get("norm_smax", 45.0)
    # is_norm is fixed to True

    if isinstance(tpcf_dict_list, dict):
        tpcf_dict_list = [tpcf_dict_list, ]
    snap1, snap2 = snap_ids[0], snap_ids[1]

    # Ensure tpcf_dict_list[0][snap1] is a list, tuple, or array
    first_snap_data = tpcf_dict_list[0][snap1]
    if not isinstance(first_snap_data, (list, tuple, np.ndarray)):
        first_snap_data = [first_snap_data]
        single_xismu = True
    else:
        single_xismu = False

    size = len(first_snap_data)
    if size == 1:
        shift = 0
    tpcf_diff_list_list = []

    if compressor is None:
        use_compressor = False
    else:
        use_compressor = True

    for i_list in range(len(tpcf_dict_list)):
        tpcf_diff_list = []
        for i in range(size):
            i_shift = i + shift
            if i_shift >= size:
                i_shift -= size
            xismu_first = tpcf_dict_list[i_list][snap1][i] if not single_xismu else tpcf_dict_list[i_list][snap1]
            xismu_second = tpcf_dict_list[i_list][snap2][i_shift] if not single_xismu else tpcf_dict_list[i_list][snap2]

            # Call get_specific_2d
            result_dict_1 = xismu_first.get_specific_2d(smin=smin, smax=smax, mumin=mumin, mumax=mumax, norm_smax=norm_smax, is_norm=True)
            result_dict_2 = xismu_second.get_specific_2d(smin=smin, smax=smax, mumin=mumin, mumax=mumax, norm_smax=norm_smax, is_norm=True)

            # Extract xis, flatten, and remove last element
            xi_temp_1 = result_dict_1["xis"].flatten()[:-1]
            xi_temp_2 = result_dict_2["xis"].flatten()[:-1]

            xi_temp_diff = (xi_temp_1 - xi_temp_2)
            tpcf_diff_list.append(xi_temp_diff)
        tpcf_diff_list_list.append(np.array(tpcf_diff_list))

    diff_array = np.concatenate(tpcf_diff_list_list, axis=1)

    # Apply PCA transformation if needed
    if use_compressor:
        from .compressor import Compressor, CompressorSeries
        if isinstance(compressor, CompressorSeries) or isinstance(compressor, Compressor):
            result_array = compressor.transform(diff_array)
        else:
            raise ValueError("compressors_list must be a list of Compressor or a single Compressor")
    else:
        result_array = diff_array

    if result_array.shape[0] == 1:
        result_array = result_array[0]

    return result_array

def create_xismu_from_pairs(DD, DR, RR, norm_DD, norm_DR, norm_RR, sedges, muedges):
    """
    Create xismu object from pair counts.

    Parameters
    ----------
    DD : ndarray
        DD pair counts (sbin, mubin).
    DR : ndarray
        DR pair counts (sbin, mubin).
    RR : ndarray
        RR pair counts (sbin, mubin).
    norm_DD : float
        Normalization factor for DD.
    norm_DR : float
        Normalization factor for DR.
    norm_RR : float
        Normalization factor for RR.
    sedges : array-like
        Separation bin edges.
    muedges : array-like
        Mu bin edges.

    Returns
    -------
    xismu_obj : xismu
        xismu object containing the correlation function.
    """
    sbin = len(sedges) - 1
    mubin = len(muedges) - 1
    smax = sedges[-1]

    # Create S and Mu meshgrid
    s_array = (sedges[1:] + sedges[:-1]) / 2.0
    mu_array = (muedges[1:] + muedges[:-1]) / 2.0
    S, Mu = np.meshgrid(s_array, mu_array, indexing="ij")

    # Create xismu object
    xismu_obj = xismu(
        smax=smax, sbin=sbin, mubin=mubin,
        DDnorm=norm_DD, DRnorm=norm_DR, RRnorm=norm_RR,
        S=S, Mu=Mu,
        DD=DD, DR=DR, RR=RR,
        set_data=True
    )

    return xismu_obj


# In-place migration of historical object archives.


_legacy_xismu_load_state = ContextVar("legacy_xismu_load_state", default=None)


def _note_legacy_xismu_reference():
    """Record an old class lookup independently of the once-only warning.

    The compatibility entry point calls this for ordinary pickle globals and
    the separate pickle streams embedded in joblib object arrays. Context-local
    state keeps concurrent or nested loads from changing one another's result.
    """
    state = _legacy_xismu_load_state.get()
    if state is not None:
        state[0] = True


def _is_xismu_instance(value):
    """Recognize the canonical class and its former tpcf pickle spelling."""
    value_type = type(value)
    return (
        isinstance(value, xismu)
        or (
            value_type.__name__ == "xismu"
            and value_type.__module__ in ("LSS_python.tpcf", "LSS_python.xismu")
        )
    )


def _migrate_xismu_graph(value, visited, migrated):
    """Retarget every reachable xismu instance without rebuilding its state."""
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)

    if _is_xismu_instance(value):
        if type(value) is not xismu:
            value.__class__ = xismu
        migrated.add(id(value))
        for attribute in vars(value).values():
            _migrate_xismu_graph(attribute, visited, migrated)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            _migrate_xismu_graph(key, visited, migrated)
            _migrate_xismu_graph(item, visited, migrated)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _migrate_xismu_graph(item, visited, migrated)
    elif isinstance(value, np.ndarray) and value.dtype.hasobject:
        for item in value.flat:
            _migrate_xismu_graph(item, visited, migrated)


def _same_migration_value(before, after, forward=None, reverse=None):
    """Compare serialized graphs, including array values and shared references."""
    if forward is None:
        forward, reverse = {}, {}
    if type(before) in (type(None), bool, int, float, complex, str, bytes):
        if type(before) is not type(after):
            return False
        return _same_scalar(before, after)
    if isinstance(before, np.generic) or isinstance(after, np.generic):
        if type(before) is not type(after):
            return False
        return _same_scalar(before, after)
    before_id, after_id = id(before), id(after)
    if before_id in forward:
        return forward[before_id] == after_id
    if after_id in reverse:
        return False
    forward[before_id] = after_id
    reverse[after_id] = before_id

    before_xismu = _is_xismu_instance(before)
    after_xismu = _is_xismu_instance(after)
    if before_xismu or after_xismu:
        return (
            before_xismu
            and after_xismu
            and type(after) is xismu
            and _same_migration_value(vars(before), vars(after), forward, reverse)
        )

    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        if not (isinstance(before, np.ndarray) and isinstance(after, np.ndarray)):
            return False
        if before.shape != after.shape or before.dtype != after.dtype:
            return False
        if before.dtype.hasobject:
            return all(
                _same_migration_value(left, right, forward, reverse)
                for left, right in zip(before.flat, after.flat)
            )
        try:
            return bool(np.array_equal(before, after, equal_nan=True))
        except (TypeError, ValueError):
            return bool(np.array_equal(before, after))

    if type(before) is not type(after):
        return False
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return False
        return all(
            _same_migration_value(before[key], after[key], forward, reverse)
            for key in before
        )
    if isinstance(before, (list, tuple)):
        return len(before) == len(after) and all(
            _same_migration_value(left, right, forward, reverse)
            for left, right in zip(before, after)
        )

    before_state = getattr(before, "__dict__", None)
    after_state = getattr(after, "__dict__", None)
    if before_state is not None or after_state is not None:
        return (
            before_state is not None
            and after_state is not None
            and _same_migration_value(before_state, after_state, forward, reverse)
        )
    try:
        result = before == after
        return bool(result) if not isinstance(result, np.ndarray) else bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _same_scalar(before, after):
    """Compare scalar values, allowing NaNs where their dtype supports them."""
    try:
        return bool(
            np.array_equal(
                np.asarray(before), np.asarray(after), equal_nan=True
            )
        )
    except (TypeError, ValueError):
        try:
            result = before == after
            return bool(result) if not isinstance(result, np.ndarray) else bool(np.all(result))
        except (TypeError, ValueError):
            return False


def _joblib_compression(filename):
    """Return a joblib compressor name from its standard leading signature."""
    with open(filename, "rb") as source:
        signature = source.read(8)
    signatures = (
        (b"\x1f\x8b", "gzip"),
        (b"BZh", "bz2"),
        (b"\xfd7zXZ", "xz"),
        (b"]\x00", "lzma"),
        (b"\x04\x22\x4d\x18", "lz4"),
    )
    for prefix, compressor in signatures:
        if signature.startswith(prefix):
            return compressor
    # Joblib's zlib wrapper uses a two-byte RFC 1950 header.
    if (
        len(signature) >= 2
        and signature[0] & 0x0F == 8
        and (signature[0] * 256 + signature[1]) % 31 == 0
    ):
        return "zlib"
    return 0


def migrate_xismu_file(filename):
    """Atomically rewrite a joblib file with canonical xismu pickle references.

    The object graph is loaded and dumped as-is, apart from changing legacy
    xismu class references. The source is replaced only after a reload confirms
    canonical class references and unchanged data, container types, and arrays.
    Returns the number of unique xismu instances rewritten, or zero when the
    archive contains no legacy class references (including current archives).
    A skipped archive is read once without dumping, reloading, or comparing it.
    """
    source_path = os.path.realpath(os.fspath(filename))
    source_stat = os.stat(source_path)
    # Old references resolve to the canonical class during loading, so instance
    # identity alone cannot tell whether an on-disk reference needs migration.
    legacy_seen = [False]
    token = _legacy_xismu_load_state.set(legacy_seen)
    try:
        original = joblib.load(source_path)
    finally:
        _legacy_xismu_load_state.reset(token)
    if not legacy_seen[0]:
        return 0

    migrated = set()
    _migrate_xismu_graph(original, set(), migrated)
    if not migrated:
        return 0

    directory = os.path.dirname(source_path) or os.curdir
    prefix = "." + os.path.basename(source_path) + ".migrate-"
    descriptor, temporary_path = tempfile.mkstemp(prefix=prefix, dir=directory)
    os.close(descriptor)
    temporary_files = {temporary_path}
    try:
        compression = _joblib_compression(source_path)
        dumped_files = joblib.dump(original, temporary_path, compress=compression)
        temporary_files.update(os.fspath(path) for path in dumped_files)
        os.chmod(temporary_path, stat.S_IMODE(source_stat.st_mode))
        reloaded = joblib.load(temporary_path)
        _validate_canonical_xismu_graph(reloaded, set())
        if not _same_migration_value(original, reloaded):
            raise ValueError("joblib migration validation found changed object data")
        os.replace(temporary_path, source_path)
        temporary_files.discard(temporary_path)
        return len(migrated)
    finally:
        temporary_prefix = os.path.basename(temporary_path)
        try:
            for entry in os.scandir(directory):
                if entry.name.startswith(temporary_prefix):
                    temporary_files.add(entry.path)
        except FileNotFoundError:
            pass
        for path in temporary_files:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _validate_canonical_xismu_graph(value, visited):
    """Reject any migrated graph containing a noncanonical xismu instance."""
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)
    if _is_xismu_instance(value):
        if type(value) is not xismu:
            raise ValueError("joblib migration left a noncanonical xismu instance")
        for item in vars(value).values():
            _validate_canonical_xismu_graph(item, visited)
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_canonical_xismu_graph(key, visited)
            _validate_canonical_xismu_graph(item, visited)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_canonical_xismu_graph(item, visited)
    elif isinstance(value, np.ndarray) and value.dtype.hasobject:
        for item in value.flat:
            _validate_canonical_xismu_graph(item, visited)
