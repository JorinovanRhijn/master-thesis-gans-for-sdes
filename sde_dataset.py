# Jorino van Rhijn
# Monte Carlo Simulation of SDEs with GANs

import numpy as np
import torch
import scipy.stats as stat
from utils import standardise


class SDEDataset():
    def __init__(self):
        # General default parameters
        self.N = 10_000
        self.N_test = 10_000
        self.N_steps = 1000

        # Set default parameters
        self.SDE = None
        self.params = dict(t=1, S0=1, S0_test=1, mu=0.05, sigma=0.2, kappa=0.5, S_bar=1, gamma=0.1, s=0)
        self.CGAN = False
        self.C = None
        self.C_test = None

    def condition_init(self, C=None, C_test=None):
        '''
        Initialise the dict with conditional parameters. Does nothing if C is not specified at all
        '''
        # Either use already specified self.C or (re)define it from method argument
        if C is not None:
            self.C = C
        if self.C is not None:
            self.CGAN = True
            if C_test is not None:
                self.C_test = C_test
            # Run two tests to prevent difficult situations later
            assert self.C_test is not None, 'Please specify a dict with test condition classes.'
            assert str(self.C.keys()) == str(self.C_test.keys()), 'The keys of the condition dict and'
            'test dict must match, also in order.'
            for key in self.C.keys():
                if hasattr(self.C[key], '__len__'):
                    assert len(self.C[key]) in [self.N, 1], 'Size of the conditional arguments must be either 1 or N.'
                    # Cast each entry into np array
                    self.C[key] = np.array(self.C[key])

            # Store the indices of the conditional dict entries
            # self.C.update(dict(indices=np.arange(0,self.N)))
            # Update the parameter set with the condition dict
            self.params.update(self.C)
            self.C_dim = len(self.C)
        else:
            self.C_test = dict()
            print('No condition dict given, providing dataset for vanilla GAN...')

    def shuffle_dataset(self):
        '''
        Shuffles the dataset class entries, making sure all parameter entries match the data after re-indexing.
        '''
        shuffle_indices = np.random.permutation(self.N)
        shuffle_indices_test = np.random.permutation(self.N_test)
        assert (self.exact is not None) & (self.exact_test is not None), 'First call generate_GBM_data or'
        'generate_CIR_data to initialise the datasets.'
        if not self.CGAN:
            self.exact = self.exact[shuffle_indices, :]
            self.exact_test = self.exact_test[shuffle_indices_test, :]
            print('Shuffled the training dataset along its first dimension.')
        else:
            for key in self.C.keys():
                if hasattr(self.C[key], '__len__'):
                    self.C[key] = self.C[key][shuffle_indices]
            self.exact = self.exact[shuffle_indices, :]
            self.exact_test = self.exact_test[shuffle_indices_test, :]
            if self.SDE == 'GBM':
                self.Z = self.Z[shuffle_indices]
                self.Z_test = self.Z_test[shuffle_indices]
            elif self.SDE == 'CIR':
                self.delta = self.delta[shuffle_indices]
                self.c_bar = self.c_bar[shuffle_indices]
                self.kappa_bar = self.kappa_bar[shuffle_indices]
            print('Shuffled the conditional dict and training dataset along its first dimension.')
            self.params.update(self.C)

    def generate_GBM_data(self, C=None, C_test=None):
        '''
        Generates variables Dataset.exact and Dataset.exact_test from the parameters within the Dataset class instance.
        '''

        self.SDE = 'GBM'
        self.condition_init(C, C_test)

        # Define the input vectors to the generator
        self.Z = standardise(torch.randn((self.N, 1)))
        self.Z_test = standardise(torch.randn((self.N_test, 1)))

        # Use these Z to get samples of S_t
        self.exact = self.get_exact_GBM_samples(Z=self.Z, params=self.params)
        self.exact_test = self.get_exact_GBM_samples(Z=self.Z_test, params={**self.params, **self.C_test})

        print(f'Stored {self.N} exactly sampled values using params {self.params} in [instance_name].Z\
        and [instance_name].exact.')
        if self.CGAN:
            print(f'Stored {self.N_test} exactly sampled values using params { {**self.params,**self.C_test} }'
                  'in [instance_name].Z_test and [instance_name].exact_test.')
        else:
            print(f'Stored {self.N_test} exactly sampled test values in [instance_name].Z_test\
                and [instance_name].exact_test.')

    def generate_CIR_data(self, C=None, C_test=None, eps=1e-20):
        '''
        Generates variables Dataset.exact and Dataset.exact_test from the parameters within the Dataset class instance.
        Also stores delta, kappa_bar and c_bar.
        '''
        self.SDE = 'CIR'
        if C is None:
            C = self.C
        if C_test is None:
            C_test = self.C_test

        self.condition_init(C, C_test)

        if not self.CGAN:
            kappa = self.params['kappa']
            gamma = self.params['gamma']
            S_bar = self.params['S_bar']
            S0 = self.params['S0']
            s = self.params['s']
            t = self.params['t']

            self.exact = self.get_exact_CIR_samples(N=self.N, params=self.params)
            self.exact_test = self.get_exact_CIR_samples(N=self.N_test, params=self.params)

            self.delta = (4*kappa*S_bar)/(gamma**2)
            self.kappa_bar = (4*kappa*S0*np.exp(-kappa*(t-s)))/(gamma**2*(1-np.exp(-kappa*(t-s))))
            self.c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(t-s)))

            print(f'Stored {self.N} exact ncx2 samples using params using params {self.params} in\
                 [instance_name].exact')
            print(f'Stored {self.N_test} exact ncx2 test samples in [instance_name].exact_test')
        else:
            self.generate_CGAN_exact_CIR_samples()

        # Prevent values being exactly 0
        self.exact += eps
        self.exact_test += eps

        # Use inverse CDF to get corresponding normal variate
        self.Z = torch.tensor(stat.norm.ppf(self.exact_cdf_CIR(params=self.params)(self.exact.view(-1).numpy())),
                              dtype=torch.float32).view(-1, 1)
        self.Z_test = torch.tensor(stat.norm.ppf(
                                                self.exact_cdf_CIR(params=self.params)
                                                (self.exact_test.view(-1).numpy())), dtype=torch.float32).view(-1, 1)

        print(f'Stored {self.N} normal samples Z in [instance_name].Z and [instance_name].Z_test')

    def get_exact_GBM_samples(self, N=None, Z=None, return_Z=False, params=None):
        '''
        S_t = get_exact_samples(self,S0,Z=None,return_Z=False) \n
        Generate exactly sampled data. If return_Z==True,
        the vector of normal random variables is returned as first argument.
        '''

        if params is None:
            params = self.params
        if N is None:
            N = self.N
        if Z is None:
            Z = standardise(torch.randn((N, 1)))

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        t = params['t']

        exact = torch.tensor(S0*np.exp((mu-0.5*sigma**2)*t+sigma*np.sqrt(t)*Z.view(-1).numpy()),
                             dtype=torch.float32).view(-1, 1)

        if return_Z:
            return Z, exact
        else:
            return exact

    def get_exact_CIR_samples(self, N=None, params=None):
        '''
        Generate exact CIR samples given an S_0 and store them in self.exact_ncx2.
        Uses the N and S0 defined in the Dataset class instance.
        '''
        if params is None:
            params = self.params
        if N is None:
            N = self.N

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']
        s = params['s']
        t = params['t']

        delta = (4*kappa*S_bar)/(gamma**2)
        kappa_bar = (4*kappa*S0*np.exp(-kappa*(t-s)))/(gamma**2*(1-np.exp(-kappa*(t-s))))
        c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(t-s)))

        exact = c_bar*np.random.noncentral_chisquare(delta, kappa_bar, size=N)
        return torch.tensor(exact, dtype=torch.float32).view(-1, 1)

    def generate_CGAN_exact_CIR_samples(self, params=None):
        '''
        *** To be redacted --- Turns out Scipy stats can handle non-scalar parameter sets out-of-the-box,
        eliminating the need for this method ***
        Method to generate a dataset of CIR samples with non-scalar parameters.
        '''
        if params is None:
            params = self.params

        kappa = params['kappa']*np.ones(self.N)
        gamma = params['gamma']*np.ones(self.N)
        S_bar = params['S_bar']*np.ones(self.N)
        S0 = params['S0']*np.ones(self.N)
        s = params['s']*np.ones(self.N)
        t = params['t']*np.ones(self.N)

        self.delta = (4*kappa*S_bar)/(gamma**2)
        self.kappa_bar = (4*kappa*S0*np.exp(-kappa*(t-s)))/(gamma**2*(1-np.exp(-kappa*(t-s))))
        self.c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(t-s)))

        exact = np.zeros(self.N)
        for i in range(self.N):
            exact[i] = self.c_bar[i]*np.random.noncentral_chisquare(self.delta[i], self.kappa_bar[i], size=1)
        self.exact = torch.tensor(exact, dtype=torch.float32).view(-1, 1)
        self.exact_test = self.get_exact_CIR_samples(N=self.N_test, params={**self.params, **self.C_test})

        print(f'Stored {self.N} exact CGAN CIR samples using params {self.params} in [instance_name].exact')
        print(f'Stored {self.N_test} exact CGAN test samples using params {{**self.params, **self.C_test}}'
              'in [instance_name].exact_test')

    def make_GBM_Euler_paths(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates paths with the Euler-Maruyama scheme.
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']

        paths = torch.zeros(N_paths, n_steps+1)

        paths[:, 0] = S0
        for n in range(n_steps):
            paths[:, n+1] = paths[:, n] + mu*paths[:, n]*dt + sigma*paths[:, n]*np.sqrt(dt)*Z[:, n]

        return paths

    def make_CIR_trunc_Euler_paths(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates paths with the Euler-Maruyama scheme using partial truncation.
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        S_bar = params['S_bar']
        kappa = params['kappa']
        S0 = params['S0']
        gamma = params['gamma']

        paths = torch.zeros(N_paths, n_steps+1)

        paths[:, 0] = S0

        for n in range(n_steps):
            paths_trunc = torch.zeros(N_paths)
            gzero_ind = paths[:, n] > 0
            paths_trunc[gzero_ind] = paths[:, n][gzero_ind]
            paths[:, n+1] = paths[:, n] + kappa*(S_bar - paths[:, n])*dt +\
                gamma*torch.sqrt(paths_trunc)*np.sqrt(dt)*Z[:, n]
        return paths

    def make_CIR_trunc_full_Euler_paths(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates paths with the Euler-Maruyama scheme using full truncation.
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        S_bar = params['S_bar']
        kappa = params['kappa']
        S0 = params['S0']
        gamma = params['gamma']

        paths = torch.zeros(N_paths, n_steps+1)

        paths[:, 0] = S0

        for n in range(n_steps):
            paths_trunc = torch.zeros(N_paths)
            gzero_ind = paths[:, n] > 0
            paths_trunc[gzero_ind] = paths[:, n][gzero_ind]
            paths[:, n+1] = paths[:, n] +\
                kappa*(S_bar - paths_trunc)*dt + gamma*torch.sqrt(paths_trunc)*np.sqrt(dt)*Z[:, n]
        return paths

    def make_CIR_refl_Euler_paths(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates paths with the reflected Euler-Maruyama scheme.
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        S_bar = params['S_bar']
        kappa = params['kappa']
        S0 = params['S0']
        gamma = params['gamma']

        paths = torch.zeros(N_paths, n_steps+1)

        paths[:, 0] = S0

        for n in range(n_steps):
            paths[:, n+1] =\
                          np.abs(paths[:, n] +
                                 kappa*(S_bar - paths[:, n])*dt + gamma*torch.sqrt(paths[:, n])*np.sqrt(dt)*Z[:, n])
        return paths

    def make_GBM_Milstein_paths(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates paths with the Euler-Maruyama scheme.
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
            print('Assuming t means final time, dt is set to t/n_steps.')
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']

        paths = torch.zeros(N_paths, n_steps+1)

        paths[:, 0] = S0

        for n in range(n_steps):
            paths[:, n+1] =\
                          paths[:, n] + mu*paths[:, n]*dt + sigma*paths[:, n]*np.sqrt(dt)*Z[:, n] +\
                          1./2*sigma**2*paths[:, n]*dt*(np.power(Z[:, n], 2)-1)
        return paths

    def make_CIR_Milstein_paths_paper(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates paths with the truncated Milstein scheme for the CIR process by (Hefter et al. (2016))
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        Z_c = Z.clone().numpy()

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']

        paths = torch.zeros(N_paths, n_steps+1)

        paths[:, 0] = S0

        m_const = 1./2*gamma*np.sqrt(dt)
        for n in range(n_steps):
            paths[:, n+1] = np.maximum(np.maximum(
                                                m_const, np.sqrt(np.maximum(m_const, paths[:, n])) +
                                                m_const*Z_c[:, n])**2+(kappa*S_bar - 1./4*gamma**2 -
                                                                       kappa*paths[:, n])*dt, 0)

        return paths

    def make_CIR_trunc_Milstein_paths(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates paths with a truncated version of the Milstein scheme.
        This is ad-hoc and performs badly on the CIR process
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
            print('Assuming t means final time, dt is set to t/n_steps.')
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']

        paths = torch.zeros(N_paths, n_steps+1)

        paths[:, 0] = S0

        for n in range(n_steps):
            paths[:, n+1] = paths[:, n] + kappa*(S_bar-paths[:, n])*dt +\
                            gamma*np.sqrt(np.maximum(dt*paths[:, n], 0))*Z[:, n] + 1./2*gamma**2*dt*(Z[:, n]**2-1)
        return paths

    def make_GBM_exact_paths(self, n_steps, dt=None, params=None, Z=None, N_paths=None):
        '''
        Method that generates exact paths for GBM.
        '''

        if params is None:
            params = self.params
        if dt is None:
            dt = params['t']
        if N_paths is None:
            N_paths = self.N
        if Z is None:
            Z = standardise(torch.randn(N_paths, n_steps))
        else:
            assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']

        exact_paths = torch.zeros((N_paths, n_steps+1))
        exact_paths[:, 0] = S0
        exact_paths[:, 1:] = S0*np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z).cumprod(axis=1)

        return exact_paths

    def generate_GBM_Milstein_approx(self, C_Mil=None, Z=None):
        '''
        Generates a Milstein approximation of the data, which it stores in Data.Mil and Data.Z_Mil.
        C_Mil:  dict with conditional parameters.
        Z:      optionally specify the random numbers used
        Note: not used for the paper (March 2021)
        '''

        params = self.params.copy()

        if C_Mil is not None:
            self.C_Mil = C_Mil

        # assert self.C_Mil is not None, 'Either C_Mil must be specified or instantiated as Data.C_Mil.'

        if self.C_Mil is not None:
            params.update(self.C_Mil)

        t = params['t']

        # Define Z_Mil, which is needed in the second discriminator
        if Z is None:
            self.Z_Mil = standardise(torch.randn((self.N, 1)))
        else:
            self.Z_Mil = Z

        # Re-use the Milstein paths method on a single step to get the approximation
        self.Mil = self.make_GBM_Milstein_paths(1, dt=t, params=params, Z=self.Z_Mil, N_paths=self.N)[:, -1].view(-1, 1)

        print(f'Stored {self.N} Milstein samples using params {params} in [instance_name].Z_Mil\
        and [instance_name].Mil.')

    def exact_pdf_GBM(self, params=None):
        '''
        Returns lambda function of the exact pdf for GBM samples given parameters in params.
        '''

        if params is None:
            params = self.params
        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        t = params['t']

        scale = np.exp(np.log(S0)+(mu-0.5*sigma**2)*(t))
        s = sigma*np.sqrt(t)
        return lambda x: stat.lognorm.pdf(x=x, scale=scale, s=s)

    def exact_cdf_GBM(self, params=None):
        '''
        Returns lambda function of the exact cdf for GBM samples given parameters in params.
        '''

        if params is None:
            params = self.params

        mu = params['mu']
        sigma = params['sigma']
        S0 = params['S0']
        t = params['t']

        scale = np.exp(np.log(S0)+(mu-0.5*sigma**2)*(t))
        s = sigma*np.sqrt(t)
        return lambda x: stat.lognorm.cdf(x=x, scale=scale, s=s)

    def exact_pdf_CIR(self, params=None):
        '''
        Returns lambda function of the exact pdf for CIR samples given parameters in params.
        '''

        if params is None:
            params = self.params

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']
        s = params['s']
        t = params['t']

        kappa_bar = (4*kappa*S0*np.exp(-kappa*(t-s)))/(gamma**2*(1-np.exp(-kappa*(t-s))))
        c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(t-s)))
        delta = (4*kappa*S_bar)/(gamma**2)
        return lambda x: stat.ncx2.pdf(x, delta, kappa_bar, scale=c_bar)

    def inv_dist_CIR(self, params=None):
        '''
        Returns lambda function of the inverse function for the CIR process given parameters in params.
        '''

        if params is None:
            params = self.params

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']
        s = params['s']
        t = params['t']

        kappa_bar = (4*kappa*S0*np.exp(-kappa*(t-s)))/(gamma**2*(1-np.exp(-kappa*(t-s))))
        c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(t-s)))
        delta = (4*kappa*S_bar)/(gamma**2)
        return lambda x: stat.ncx2.ppf(x, delta, kappa_bar, scale=c_bar)

    def exact_cdf_CIR(self, params=None):
        '''
        Returns lambda function of the exact cdf for CIR samples given parameters in params.
        '''

        if params is None:
            params = self.params

        kappa = params['kappa']
        gamma = params['gamma']
        S_bar = params['S_bar']
        S0 = params['S0']
        s = params['s']
        t = params['t']

        kappa_bar = (4*kappa*S0*np.exp(-kappa*(t-s)))/(gamma**2*(1-np.exp(-kappa*(t-s))))
        c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(t-s)))
        delta = (4*kappa*S_bar)/(gamma**2)
        return lambda x: stat.ncx2.cdf(x, delta, kappa_bar, scale=c_bar)
