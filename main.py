# Jorino van Rhijn
# Monte Carlo Simulation of SDEs with GANs

from itertools import cycle
import torch
import time
import os
import numpy as np
import os.path as pt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import statsmodels.distributions as smd
import scipy.stats as stat

from dataclasses import asdict
from data_types import Config
from data import DatasetBase
from sample import preprocess, postprocess
from utils import count_layers, append_gradients, dict_to_tensor,\
    make_test_tensor, pickle_it, select_device, get_plot_bounds
from sample import input_sample
from train_step import step_handler
from nets import Generator, Discriminator
from presets import load_preset
from config.load_config import load_config


config: Config = load_config()
if not pt.exists(config.meta_parameters.default_dir):
    os.mkdir(config.meta_parameters.default_dir)

# Define all metaparameters in dict

# Use GPU if CUDA set to True and GPU is available
DEVICE = select_device() if config.meta_parameters.enable_cuda else torch.device("cpu")

torch.manual_seed(config.meta_parameters.seed)
np.random.seed(seed=config.meta_parameters.seed)


def train_GAN(netD: Discriminator, netG: Generator, dataset: DatasetBase):
    '''
    Training loop: train_GAN(netD, netG, data, meta)
    Inspired by several tricks from DCGAN PyTorch tutorial.
    '''

    # -------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------

    real_label = 1
    fake_label = 0
    GANloss = nn.BCELoss()

    GAN_step = step_handler(supervised_bool=config.meta_parameters.supervised,
                            CGAN_bool=config.meta_parameters.conditional_gan)

    # Initialise lists for logging
    D_losses = []
    G_losses = []
    times = []
    wasses = []
    ksstat = []
    D_grads = []
    G_grads = []

    if not pt.exists(pt.join(config.meta_parameters.default_dir, 'training', '')):
        os.mkdir(pt.join(config.meta_parameters.default_dir, 'training', ''))
    # train_analysis = CGANalysis(dataset,
    #                             netD,
    #                             netG,
    #                             save_all_figs=config.meta_parameters.save_figs,
    #                             results_path=pt.join(config.meta_parameters.default_dir, 'training', ''),
    #                             proc_type=config.meta_parameters.proc_type,
    #                             eps=config.net_parameters.eps,
    #                             supervised=config.meta_parameters.supervised,
    #                             device=DEVICE)

    # train_analysis.format = 'png'

    (Z_train, X_train), (_, X_test) = dataset.generate_train_test()
    X_train = preprocess(X_train,
                         X_prev=dataset.params['S0'],
                         proc_type=config.meta_parameters.proc_type,
                         S_ref=config.meta_parameters.S_ref,
                         eps=config.meta_parameters.eps)

    if dataset.condition_ranges is not None:
        C_tensors = dict_to_tensor(dataset.condition_dict)
        C_test = make_test_tensor(config.test_parameters.test_condition, dataset.n_test)
    else:
        C_test = None

    # Pre-allocate an empty array for each layer to store the norm
    for _ in range(count_layers(netD)):
        D_grads.append([])

    for _ in range(count_layers(netG)):
        G_grads.append([])

    # Initialise counters
    itervec = []
    iters = 0
    plot_iter = 0

    # Get the amount of batches implied by training set size and batch_size
    n_batches = dataset.n // config.train_parameters.batch_size

    optG = optim.Adam(netG.parameters(),
                      lr=config.train_parameters.lr_G,
                      betas=(config.train_parameters.beta1,
                      config.train_parameters.beta2))
    optD = optim.Adam(netD.parameters(),
                      lr=config.train_parameters.lr_D,
                      betas=(config.train_parameters.beta1,
                      config.train_parameters.beta2))

    # -------------------------------------------------------
    # Start training loop
    # -------------------------------------------------------

    for epoch in range(config.train_parameters.epochs):
        tick0 = time.time()
        for i in range(n_batches):
            if iters % config.train_parameters.cut_lr_every == 0:
                optG.param_groups[0]['lr'] = optG.param_groups[0]['lr'] / config.train_parameters.c_lr

            # Sample random minibatch from training set with replacement
            indices = np.array((np.random.rand(config.train_parameters.batch_size)*dataset.n), dtype=int)
            # Uncomment to sample minibatch from training set without replacement
            # indices = np.arange(i*b_size,(i+1)*b_size)

            # Get data batch based on indices
            X_i = X_train[indices, :].to(DEVICE)
            C_i = C_tensors[indices, :].to(DEVICE) if config.meta_parameters.conditional_gan else None
            Z_i = Z_train[indices, :].to(DEVICE) if config.meta_parameters.supervised else None

            # -------------------------------------------------------
            # GAN training step
            # -------------------------------------------------------

            errD, errG, D_x, D_G_z1, D_G_z2 = GAN_step(netD,
                                                       netG,
                                                       optD,
                                                       optG,
                                                       GANloss,
                                                       DEVICE,
                                                       X_i=X_i,
                                                       C_i=C_i,
                                                       Z_i=Z_i,
                                                       real_label=real_label,
                                                       fake_label=fake_label,
                                                       n_D=config.train_parameters.n_D)

            # -------------------------------------------------------
            # End of train step, start logging
            # -------------------------------------------------------

            # Output training stats
            if (iters % 100 == 0) and (i % config.train_parameters.batch_size) == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config.train_parameters.epochs, i, dataset.n // config.train_parameters.batch_size,
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                input_G = input_sample(dataset.n_test, C=C_test, device=DEVICE)
                fake = postprocess(netG(input_G).detach().view(-1),
                                   {**dataset.params, **dataset.test_params}['S0'],
                                   proc_type=config.meta_parameters.proc_type,
                                   S_ref=torch.tensor(dataset.params['S_bar'],
                                   device=DEVICE,
                                   dtype=torch.float32),
                                   eps=config.net_parameters.eps).cpu().view(-1).numpy()
                ecdf_fake = smd.ECDF(fake)
                ecdf_test = smd.ECDF(X_test.view(-1))
                # x = np.linspace(1e-5, 3, 1000)  # plotting vector
                x_min, x_max = get_plot_bounds(fake)
                x = np.linspace(x_min, x_max, 1000)

                # Infinity norm with ECDF on test dataset.(Kolmogorov-Smirnov statistic)
                ksstat.append(np.max(np.abs(ecdf_fake(x)-ecdf_test(x))))
                # ksstat.append(stat.kstest(fake,cdf=cdf_test),alternative='two-sided')[0])

                # 1D Wasserstein distance as implemented in Scipy stats package
                wasses.append(stat.wasserstein_distance(fake, X_test.view(-1)))

                # Keep track of the L1 norm of the gradients in each layer
                append_gradients(netD, netG, D_grads, G_grads)

                itervec.append(iters)

            # Update the generated dataset.in analysis instance
            if ((iters % config.meta_parameters.plot_interval == 0) and (config.meta_parameters.save_iter_plot)):
                # Update network references for inference
                # train_analysis.save_iter_plot(iters, params=dataset.params, D=netD, G=netG)
                plot_iter += 1

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        tick1 = time.time()
        times.append(tick1-tick0)

    # Get range of training parameters if CGAN
    if dataset.condition_ranges is not None:
        C_ranges = dict()
        for key in list(dataset.condition_ranges.keys()):
            C_ranges[key] = (min(dataset.condition_ranges[key]), max(dataset.condition_ranges[key]))

    # Create dict for output log
    output_dict = dict(
        iterations=np.arange(1, iters+1),
        iters=itervec,
        D_losses=D_losses,
        G_losses=G_losses,
        Wass_dist_test=wasses,
        KS_stat_test=ksstat,
        D_layers=[count_layers(netD)],
        G_layers=[count_layers(netG)],
        final_lr_G=[optG.param_groups[0]['lr']],
        final_lr_D=[optD.param_groups[0]['lr']],
        total_time=[np.sum(times)],
        train_condition=list(dataset.condition_dict.keys()) if dataset.condition_dict is not None else None,
        train_condition_ranges=[str(C_ranges)] if dataset.condition_dict is not None else None,
        test_condition=[str(dataset.test_params)] if dataset.condition_dict is not None else None,
        params_names=list(dataset.params.keys()),
        params=list(dataset.params.values()),
        SDE=[dataset.SDE],
        )

    # Add metaparameters to output log
    output_dict = {**asdict(config.meta_parameters), **output_dict}

    for k in range(len(G_grads)):
        dict_entry = dict()
        dict_entry['G_grad_layer_%d' % k] = G_grads[k]
        output_dict.update(dict_entry)

    for k in range(len(D_grads)):
        dict_entry = dict()
        dict_entry['D_grad_layer_%d' % k] = D_grads[k]
        output_dict.update(dict_entry)

    # Convert to Pandas DataFrame
    results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in output_dict.items()]))

    pd.concat((results_df, pd.DataFrame(dataset.test_params, index=[0])), axis=1, ignore_index=True)

    results_df = pd.concat([results_df, pd.DataFrame(G_grads[k], columns=['GradsG_L%d' % k])], axis=1, sort=False)
    for k in range(len(D_grads)):
        results_df = pd.concat([results_df, pd.DataFrame(D_grads[k], columns=['GradsD_L%d' % k])], axis=1, sort=False)

    print('----- Training complete -----')

    return output_dict, results_df


def main():
    # Supervised GAN?
    cycle_supervised = [False, True]
    cycle_names = ['vanilla', 'supervised']

    results_path = config.meta_parameters.default_dir

    for i in range(len(cycle_supervised)):
        # Reset the seed at each iteration for equal initalisation of the nets
        torch.manual_seed(config.meta_parameters.seed)
        np.random.seed(seed=config.meta_parameters.seed)

        # Make folder for each run of the training loop
        if not pt.exists(pt.join(results_path, cycle_names[i])):
            os.mkdir(pt.join(results_path, cycle_names[i]))

        # Modify training conditions in loop
        config.meta_parameters.supervised = cycle_supervised[i]

        # Override the default n_D, the amount of training steps of D per G training step if vanilla GAN.
        config.train_parameters.n_D = 1 if config.meta_parameters.supervised else config.train_parameters.n_D

        # Make the dataset and initialise the GAN
        dataset = load_preset(config.meta_parameters.preset,
                              n_train=config.train_parameters.n_train,
                              n_test=config.test_parameters.n_test)

        c_dim = 0 if dataset.condition_ranges is None else len(dataset.condition_ranges)
        netG = Generator(c_dim=c_dim).to(DEVICE)
        netG.eps = config.net_parameters.eps
        c_dim_discr = c_dim + 1 if config.meta_parameters.supervised else c_dim

        netD = Discriminator(c_dim=c_dim_discr,
                             negative_slope=config.net_parameters.negative_slope,
                             hidden_dim=config.net_parameters.hidden_dim,
                             activation=config.net_parameters.activation).to(DEVICE)

        # Traing the GAN
        output_dict, results_df = train_GAN(netD, netG, dataset)

        # Store results
        netG_dir = pt.join(results_path, cycle_names[i], 'netG.pth')
        netD_dir = pt.join(results_path, cycle_names[i], 'netD.pth')
        torch.save(netG.state_dict(), netG_dir)
        print('Saved Generator in %s' % netG_dir)
        torch.save(netD.state_dict(), netD_dir)
        print('Saved Discriminator in %s' % netD_dir)

        if config.meta_parameters.save_log_dict:
            results_df.to_csv(pt.join(results_path, cycle_names[i], 'train_log.csv'), index=False, header=True)
            if config.meta_parameters.save_log_dict:
                log_path = pt.join(results_path, cycle_names[i], 'train_log.pkl')
                pickle_it(output_dict, log_path)
            meta_path = pt.join(results_path, cycle_names[i], 'metadata.pkl')
            pickle_it(config, meta_path)
            print('Saved logs in ' + results_path + cycle_names[i])

    print('----- Experiment finished -----')


if __name__ == '__main__':
    main()
