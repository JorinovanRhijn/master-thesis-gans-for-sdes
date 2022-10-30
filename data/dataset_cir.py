import numpy as np
import torch
import scipy.stats as stat
from utils import standardise, make_condition_cart_product
from typing import Tuple, Dict, Union
from data import DatasetBase
from data_types import DistributionType, SchemeType


class CIRDataset(DatasetBase):
    DEFAULT_PARAMS = dict(dt=1, S0=0.1, S0_test=0.1, kappa=0.1, S_bar=0.1, gamma=0.1)

    def __init__(self,
                 n: int = 10_000,
                 n_test: int = 10_000,
                 params: Dict[str, Union[float, int]] = None,
                 test_params: Dict[str, Union[float, int]] = None,
                 condition_ranges: Dict[str, Union[float, int, np.array]] = None,
                 ):
        self.n = n
        self.n_test = n_test
        self.SDE = 'CIR'
        self.params = params if params is not None else self.DEFAULT_PARAMS
        self.test_params = test_params if test_params is not None else dict(S0=self.DEFAULT_PARAMS['S0'])
        self.condition_ranges = condition_ranges
        if condition_ranges is not None:
            condition_dict = make_condition_cart_product(condition_ranges, self.n)
        else:
            condition_dict = None
        self.condition_init(condition_dict)

    @staticmethod
    def _get_distribution(params: dict, dist_type: DistributionType):
        '''
        Returns lambda function of the exact pdf for CIR samples given parameters in params.
        '''

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']
        dt = params['dt']

        kappa_bar = (4*kappa*S0*np.exp(-kappa*(dt)))/(gamma**2*(1-np.exp(-kappa*(dt))))
        c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(dt)))
        delta = (4*kappa*S_bar)/(gamma**2)

        if dist_type is DistributionType.PDF:
            return lambda x: stat.ncx2.pdf(x, delta, kappa_bar, scale=c_bar)
        elif dist_type is DistributionType.CDF:
            return lambda x: stat.ncx2.cdf(x, delta, kappa_bar, scale=c_bar)
        elif dist_type is DistributionType.PPF:
            return lambda x: stat.ncx2.ppf(x, delta, kappa_bar, scale=c_bar)
        else:
            raise ValueError

    def pdf(self, params: dict = None):
        if params is None:
            params = self.params
        return self._get_distribution(params, dist_type=DistributionType.PDF)

    def cdf(self, params: dict = None):
        if params is None:
            params = self.params
        return self._get_distribution(params, dist_type=DistributionType.CDF)

    def ppf(self, params: dict = None):
        if params is None:
            params = self.params
        return self._get_distribution(params, dist_type=DistributionType.PPF)

    @staticmethod
    def _path_step(path_prev, kappa, gamma, S_bar, dt, Z, scheme: SchemeType):
        if scheme is SchemeType.CIR_TRUNC_EULER:
            """
            Partially truncated Euler
            """
            paths_trunc = torch.zeros(len(path_prev))
            gzero_ind = path_prev > 0
            paths_trunc[gzero_ind] = path_prev[gzero_ind]
            return path_prev + kappa*(S_bar - path_prev)*dt +\
                gamma*torch.sqrt(paths_trunc)*np.sqrt(dt)*Z
        elif scheme is SchemeType.CIR_TRUNC_EULER_FULL:
            """
            Fully truncated Euler
            """
            paths_trunc = torch.zeros(len(path_prev))
            gzero_ind = path_prev > 0
            paths_trunc[gzero_ind] = path_prev[gzero_ind]
            return path_prev + kappa*(S_bar - paths_trunc)*dt +\
                gamma*torch.sqrt(paths_trunc)*np.sqrt(dt)*Z
        elif scheme is SchemeType.CIR_REFL_EULER:
            """
            Reflected Euler
            """
            return np.abs(path_prev + kappa*(S_bar - path_prev)*dt + gamma*torch.sqrt(path_prev)*np.sqrt(dt)*Z)
        elif scheme is SchemeType.CIR_MILSTEIN_TRUNC:
            """
            Truncated version of the Milstein scheme.
            This is ad-hoc and performs badly on the CIR process
            """
            return path_prev + kappa*(S_bar-path_prev)*dt + \
                gamma*np.sqrt(np.maximum(dt*path_prev, 0))*Z + 1./2*gamma**2*dt*(Z**2-1)
        elif scheme is SchemeType.CIR_MILSTEIN_HEFTER:
            """
            Truncated Milstein scheme for the CIR process by (Hefter et al. (2016))
            """
            m_const = 1./2*gamma*np.sqrt(dt)
            return np.maximum(np.maximum(
                                        m_const, np.sqrt(np.maximum(m_const, path_prev)) +
                                        m_const*Z)**2+(kappa*S_bar - 1./4*gamma**2 - kappa*path_prev)*dt, 0)
        else:
            raise ValueError

    def condition_init(self, condition_dict):
        '''
        Initialise the dict with conditional parameters. Does nothing if C is not specified at all
        '''
        if condition_dict is not None:
            assert self.test_params is not None, "Must specify a test condition."
            assert str(condition_dict.keys()) == str(self.test_params.keys()),\
                'The keys of the condition dict and test dict must match, also in order.'
            for key in condition_dict.keys():
                if hasattr(condition_dict[key], '__len__'):
                    assert len(condition_dict[key]) in [self.n, 1],\
                        'Size of the conditional arguments must be either 1 or N.'
                    # Cast each entry into np array
                    condition_dict[key] = np.array(condition_dict[key])
            # Update the parameter set with the condition dict
            self.params.update(condition_dict)
            self.condition_dict = condition_dict

    def sample_exact(self,
                     n: int = None,
                     params: dict = None):
        '''
        Generate exact CIR samples given an S_0 and store them in self.exact_ncx2.
        Uses the N and S0 defined in the Dataset class instance.
        '''
        if params is None:
            params = self.params
        if n is None:
            n = self.n

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']
        dt = params['dt']

        delta = (4*kappa*S_bar)/(gamma**2)
        kappa_bar = (4*kappa*S0*np.exp(-kappa*(dt)))/(gamma**2*(1-np.exp(-kappa*(dt))))
        c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(dt)))

        exact = c_bar*np.random.noncentral_chisquare(delta, kappa_bar, size=n)
        return torch.tensor(exact, dtype=torch.float32).view(-1, 1)

    def make_paths(self,
                   scheme: SchemeType,
                   n_steps,
                   params=None,
                   Z=None,
                   n_paths=None,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Method that generates paths with the scheme specified in path_type.
        '''

        if params is None:
            params = self.params
        if n_paths is None:
            n_paths = self.n
        if Z is None:
            Z = standardise(torch.randn(n_paths, n_steps, dtype=torch.float32))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        S_bar = params['S_bar']
        kappa = params['kappa']
        S0 = params['S0']
        gamma = params['gamma']
        dt = params['dt']

        paths = torch.zeros(n_paths, n_steps+1)
        paths[:, 0] = S0

        for n in range(n_steps):
            paths[:, n+1] = self._path_step(paths[:, n], kappa, gamma, S_bar, dt, Z[:, n], scheme=scheme)
        return Z, paths

    def generate_train_test(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Generate a train and test set, returns two tuples of the form (prior_samples, exact_variates)
        '''
        exact = self.sample_exact(n=self.n, params=self.params)
        test_params = {**self.params, **self.test_params}
        exact_test = self.sample_exact(n=self.n_test, params=test_params)

        cdf = self._get_distribution(self.params, dist_type=DistributionType.CDF)
        cdf_test = self._get_distribution(test_params, dist_type=DistributionType.CDF)

        Z = torch.tensor(stat.norm.ppf(cdf(exact.view(-1).numpy())), dtype=torch.float32).view(-1, 1)
        Z_test = torch.tensor(stat.norm.ppf(cdf_test(exact_test.view(-1).numpy())), dtype=torch.float32).view(-1, 1)

        return (Z, exact), (Z_test, exact_test)
