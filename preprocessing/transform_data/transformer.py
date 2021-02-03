from itertools import chain, combinations
import numbers
import warnings
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize
from scipy.special import boxcox

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.sparsefuncs_fast import (inplace_csr_row_normalize_l1,
                                      inplace_csr_row_normalize_l2)
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                 mean_variance_axis, incr_mean_variance_axis,
                                 min_max_axis)
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES, _deprecate_positional_args)

# from ._csr_polynomial_expansion import _csr_polynomial_expansion
#
# from ._encoders import OneHotEncoder

BOUNDS_THRESHOLD = 1e-7

"""
Edit of sklearn yeo-johnson transformation
"""


class PowerTransformer(TransformerMixin, BaseEstimator):
    """Apply a power transform featurewise to make data more Gaussian-like.
    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.
    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.
    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.
    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.
    Read more in the :ref:`User Guide <preprocessing_transformer>`.
    .. versionadded:: 0.20
    Parameters
    ----------
    method : str, (default='yeo-johnson')
        The power transform method. Available methods are:
        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values
    standardize : boolean, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.
    copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.
    Attributes
    ----------
    lambdas_ : array of float, shape (n_features,)
        The parameters of the power transformation for the selected features.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PowerTransformer
    >>> pt = PowerTransformer()
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(pt.fit(data))
    PowerTransformer()
    >>> print(pt.lambdas_)
    [ 1.386... -3.100...]
    >>> print(pt.transform(data))
    [[-1.316... -0.707...]
     [ 0.209... -0.707...]
     [ 1.106...  1.414...]]
    See also
    --------
    power_transform : Equivalent function without the estimator API.
    QuantileTransformer : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.
    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    References
    ----------
    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).
    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).
    """
    @_deprecate_positional_args
    def __init__(self, method='yeo-johnson', *, standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        """Estimate the optimal parameter lambda for each feature.
        The optimal lambda parameter for minimizing skewness is estimated on
        each feature independently using maximum likelihood.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters.
        y : Ignored
        Returns
        -------
        self : object
        """
        self._fit(X, y=y, force_transform=False)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X, y, force_transform=True)

    def _fit(self, X, y=None, force_transform=False):
        X = self._check_input(X, in_fit=True, check_positive=True,
                              check_method=True)

        if not self.copy and not force_transform:  # if call from fit()
            X = X.copy()  # force copy so that fit does not change X inplace

        optim_function = {'box-cox': self._box_cox_optimize,
                          'yeo-johnson': self._yeo_johnson_optimize
                          }[self.method]
        with np.errstate(invalid='ignore'):  # hide NaN warnings
            self.lambdas_ = np.array([optim_function(col) for col in X.T])

        if self.standardize or force_transform:
            transform_function = {'box-cox': boxcox,
                                  'yeo-johnson': self._yeo_johnson_transform
                                  }[self.method]
            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid='ignore'):  # hide NaN warnings
                    X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            self._scaler = StandardScaler(copy=False)
            if force_transform:
                X = self._scaler.fit_transform(X)
            else:
                self._scaler.fit(X)

        return X

    def transform(self, X):
        """Apply the power transform to each feature using the fitted lambdas.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to be transformed using a power transformation.
        Returns
        -------
        X_trans : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_positive=True,
                              check_shape=True)

        transform_function = {'box-cox': boxcox,
                              'yeo-johnson': self._yeo_johnson_transform
                              }[self.method]
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid='ignore'):  # hide NaN warnings
                X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            X = self._scaler.transform(X)

        return X

    def inverse_transform(self, X):
        """Apply the inverse power transformation using the fitted lambdas.
        The inverse of the Box-Cox transformation is given by::
            if lambda_ == 0:
                X = exp(X_trans)
            else:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_)
        The inverse of the Yeo-Johnson transformation is given by::
            if X >= 0 and lambda_ == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda_ != 0:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
            elif X < 0 and lambda_ != 2:
                X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
            elif X < 0 and lambda_ == 2:
                X = 1 - exp(-X_trans)
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The transformed data.
        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            The original data
        """
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        if self.standardize:
            X = self._scaler.inverse_transform(X)

        inv_fun = {'box-cox': self._box_cox_inverse_tranform,
                   'yeo-johnson': self._yeo_johnson_inverse_transform
                   }[self.method]
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid='ignore'):  # hide NaN warnings
                X[:, i] = inv_fun(X[:, i], lmbda)

        return X

    def _box_cox_inverse_tranform(self, x, lmbda):
        """Return inverse-transformed input x following Box-Cox inverse
        transform with parameter lambda.
        """
        if lmbda == 0:
            x_inv = np.exp(x)
        else:
            x_inv = (x * lmbda + 1) ** (1 / lmbda)

        return x_inv

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        """
        x_inv = np.zeros_like(x)
        pos = x >= 0

        # when x >= 0
        if abs(lmbda) < np.spacing(1.):
            x_inv[pos] = np.exp(x[pos]) - 1
        else:  # lmbda != 0
            x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.):
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1,
                                       1 / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-x[~pos])

        return x_inv

    def _yeo_johnson_transform(self, x, lmbda):
        """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """

        out = np.zeros_like(x)
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(lmbda) < np.spacing(1.):
            out[pos] = np.log1p(x[pos])
        else:  # lmbda != 0
            out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.):
            out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:  # lmbda == 2
            out[~pos] = -np.log1p(-x[~pos])

        return out

    def _box_cox_optimize(self, x):
        """Find and return optimal lambda parameter of the Box-Cox transform by
        MLE, for observed data x.
        We here use scipy builtins which uses the brent optimizer.
        """
        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        _, lmbda = stats.boxcox(x[~np.isnan(x)], lmbda=None)

        return lmbda

    def _yeo_johnson_optimize(self, x):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed data x.
        Like for Box-Cox, MLE is done via the brent optimizer.
        """

        def _neg_log_likelihood(lmbda):
            """Return the negative log likelihood of the observed data x as a
            function of lambda."""
            x_trans = self._yeo_johnson_transform(x, lmbda)
            n_samples = x.shape[0]

            # added
            variance = x_trans.var()
            if variance == 0:
                return np.inf
            # done

            #loglike = -n_samples / 2 * np.log(x_trans.var())
            loglike = -n_samples / 2 * np.log(variance)
            loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()

            return -loglike

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        # choosing bracket -2, 2 like for boxcox
        return optimize.brent(_neg_log_likelihood, brack=(-5, 5))

    def _check_input(self, X, in_fit, check_positive=False, check_shape=False,
                     check_method=False):
        """Validate the input before fit and transform.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        check_positive : bool
            If True, check that all data is positive and non-zero (only if
            ``self.method=='box-cox'``).
        check_shape : bool
            If True, check that n_features matches the length of self.lambdas_
        check_method : bool
            If True, check that the transformation method is valid.
        """
        X = self._validate_data(X, ensure_2d=True, dtype=FLOAT_DTYPES,
                                copy=self.copy, force_all_finite='allow-nan')

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings(
                'ignore', r'All-NaN (slice|axis) encountered')
            if (check_positive and self.method == 'box-cox' and
                    np.nanmin(X) <= 0):
                raise ValueError("The Box-Cox transformation can only be "
                                 "applied to strictly positive data")

        if check_shape and not X.shape[1] == len(self.lambdas_):
            raise ValueError("Input data has a different number of features "
                             "than fitting data. Should have {n}, data has {m}"
                             .format(n=len(self.lambdas_), m=X.shape[1]))

        valid_methods = ('box-cox', 'yeo-johnson')
        if check_method and self.method not in valid_methods:
            raise ValueError("'method' must be one of {}, "
                             "got {} instead."
                             .format(valid_methods, self.method))

        return X

    def _more_tags(self):
        return {'allow_nan': True}