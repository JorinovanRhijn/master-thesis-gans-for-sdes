import numpy as np
import torch
import scipy.stats as stat
from utils import standardise
from typing import Tuple, Dict, Any
from data import DatasetBase
from data_types import DistributionType


class GBMDataset(DatasetBase):
    def __init__(self,
                 params: Dict[str, Any] = None,
                 condition_ranges: Dict[str, Any] = None,
                 condition_test: Dict[str, Any] = None,
                 ):
        self.SDE = 'GBM'
        _params_default = dict(dt=1, S0=1, S0_test=1, mu=0.05, sigma=0.2)
        self.params = params if params is not None else _params_default
        self.condition_ranges = condition_ranges if condition_ranges is not None else dict()
        self.condition_dict = self.make_condition_cart_product(self.condition_ranges)
        self.condition_dict_test = condition_test
        self.condition_init(self.condition_dict, self.condition_dict_test)

    def sample_exact(self,
                     n: int = None,
                     Z: torch.Tensor = None,
                     params: dict = None):
        '''
        S_t = get_exact_samples(self,S0,Z=None,return_Z=False) \n
        Generate exactly sampled data. If return_Z,
        the vector of normal random variables is returned as first argument.
        '''
        _, params, Z, _ = self._init_params(1, params, Z, n)

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        dt = params['dt']

        exact = torch.tensor(S0*np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z.view(-1).numpy()),
                             dtype=torch.float32).view(-1, 1)

        return Z, exact

    def _init_params(self, n_steps, params, Z, n_paths):
        if params is None:
            params = self.params
        if n_paths is None:
            n_paths = self.n
        if Z is None:
            Z = standardise(torch.randn(n_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'
        return n_steps, params, Z, n_paths

    def euler_paths(self, n_steps, dt=None, params=None, Z=None, n_paths=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Method that generates paths with the Euler-Maruyama scheme.
        '''
        n_steps, params, Z, n_paths = self._init_params(n_steps, params, Z, n_paths)

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        dt = params['dt']

        paths = torch.zeros(n_paths, n_steps+1)
        paths[:, 0] = S0

        for n in range(n_steps):
            paths[:, n+1] = paths[:, n] + mu*paths[:, n]*dt + sigma*paths[:, n]*np.sqrt(dt)*Z[:, n]

        return Z, paths

    def milstein_paths(self, n_steps, params=None, Z=None, n_paths=None):
        '''
        Method that generates paths with the Milstein scheme.
        '''
        n_steps, params, Z, n_paths = self._init_params(n_steps, params, Z, n_paths)

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        dt = params['dt']

        paths = torch.zeros(n_paths, n_steps+1)
        paths[:, 0] = S0

        for n in range(n_steps):
            paths[:, n+1] =\
                          paths[:, n] + mu*paths[:, n]*dt + sigma*paths[:, n]*np.sqrt(dt)*Z[:, n] +\
                          1./2*sigma**2*paths[:, n]*dt*(np.power(Z[:, n], 2)-1)
        return paths

    @staticmethod
    def _get_distribution(params: dict, dist_type: DistributionType):
        '''
        Returns lambda function of the exact pdf for GBM samples given parameters in params.
        '''
        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        dt = params['dt']

        scale = np.exp(np.log(S0)+(mu-0.5*sigma**2)*(dt))
        s = sigma*np.sqrt(dt)
        if dist_type is DistributionType.PDF:
            return lambda x: stat.lognorm.pdf(x=x, scale=scale, s=s)
        elif dist_type is DistributionType.CDF:
            return lambda x: stat.lognorm.cdf(x=x, scale=scale, s=s)
        elif dist_type is DistributionType.PPF:
            return lambda x: stat.lognorm.ppf(x=x, scale=scale, s=s)
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

    def generate(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Generate a train and test set, returns two tuples of the form (prior_samples, output_samples)
        '''
        Z = standardise(torch.randn((self.n, 1)))
        Z_test = standardise(torch.randn((self.n_test, 1)))

        exact = self.sample_exact(Z=Z, params=self.params)
        exact_test = self.sample_exact(Z=Z_test, params={**self.params, **self.condition_dict_test})

        return (Z, exact), (Z_test, exact_test)


# class CIRDataset(DatasetBase):
#     def __init__(self):
#         self.N = 10_000
#         self.N_test = 10_000
#         self.N_steps = 1000
#         self.SDE = 'CIR'
#         self.params = dict(t=1, S0=1, S0_test=1, kappa=0.5, S_bar=1, gamma=0.1, s=0)
#         self.C = None
#         self.C_test = None
