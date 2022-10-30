import numpy as np
import torch
import scipy.stats as stat
from utils import standardise, make_condition_cart_product
from typing import Tuple, Dict, Union
from data import DatasetBase
from data_types import DistributionType, SchemeType


class GBMDataset(DatasetBase):
    DEFAULT_PARAMS = dict(dt=1, S0=1, S0_test=1, mu=0.05, sigma=0.2)

    def __init__(self,
                 n: int = 10_000,
                 n_test: int = 10_000,
                 params: Dict[str, Union[float, int]] = None,
                 test_params: Dict[str, Union[float, int]] = None,
                 condition_ranges: Dict[str, Union[float, int, np.array]] = None,
                 ):
        self.n = n
        self.n_test = n_test
        self.SDE = 'GBM'
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

    @staticmethod
    def _path_step(path_prev, mu, sigma, dt, Z, scheme: SchemeType):
        if scheme is SchemeType.GBM_EULER:
            return path_prev + mu*path_prev*dt + sigma*path_prev*np.sqrt(dt)*Z
        elif scheme is SchemeType.GBM_MILSTEIN:
            return path_prev + mu*path_prev*dt + sigma*path_prev*np.sqrt(dt)*Z +\
             1./2*sigma**2*path_prev*dt*(np.power(Z, 2)-1)
        else:
            return ValueError

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
                     Z: torch.Tensor = None,
                     params: dict = None):
        '''
        S_t = get_exact_samples(self,S0,Z=None,return_Z=False)
        Generate exactly sampled data.
        '''

        if params is None:
            params = self.params
        if n is None:
            n = self.n
        if Z is None:
            Z = standardise(torch.randn((n, 1), dtype=torch.float32))

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        dt = params['dt']

        exact = torch.tensor(S0*np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z.view(-1).numpy()),
                             dtype=torch.float32).view(-1, 1)
        return Z, exact

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
            Z = standardise(torch.randn(n_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        dt = params['dt']

        paths = torch.zeros(n_paths, n_steps+1)
        paths[:, 0] = S0

        for n in range(n_steps):
            paths[:, n+1] = self._path_step(paths[:, n], mu, sigma, dt, Z[:, n], scheme=scheme)
        return Z, paths

    def generate_train_test(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Generate a train and test set, returns two tuples of the form (prior_samples, exact_variates)
        '''
        Z = standardise(torch.randn((self.n, 1), dtype=torch.float32))
        Z_test = standardise(torch.randn((self.n_test, 1), dtype=torch.float32))

        exact = self.sample_exact(Z=Z, params=self.params)
        exact_test = self.sample_exact(Z=Z_test, params={**self.params, **self.test_params})

        return (Z, exact), (Z_test, exact_test)
