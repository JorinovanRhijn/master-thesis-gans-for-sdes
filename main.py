# Jorino van Rhijn 
# Monte Carlo Simulation of SDEs with GANs 

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import torch
import time 
import os
import os.path as pt
import torch.nn as nn
import torch.optim as optim
import statsmodels.distributions as smd
import scipy.stats as stat
from scipy.stats import entropy
import pandas as pd

# Import custom modules 
from CGANalysis import CGANalysis
from GANutils import input_sample, count_layers, append_gradients, dict_to_tensor, make_test_tensor, preprocess, postprocess,\
pickle_it, unpickle
from SDE_Dataset import SDE_Dataset, load_preset
from train_step import step_handler
from nets import Generator, Discriminator 

# GLOBAL VARIABLES

# Manage directory to save results 
DIR = pt.dirname(pt.abspath(__file__))
DEFAULT_DIR = pt.join(DIR,'train_results')
if not pt.exists(DEFAULT_DIR):
    os.mkdir(DEFAULT_DIR)

# Initialise the RNG 
SEED = 0
CUDA = True 

# Use GPU if CUDA set to True and GPU is available 
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() & CUDA) else 'cpu')

# Define all metaparameters in dict 
META = dict(#
    c_lr=1.05, # factor by which the learning rate is divided every cut_lr_evert iterations 
    cut_lr_every = 500, 
    epochs=200,
    eps=1e-20, # small number added to logreturns and generator output to prevent exactly reaching 0
    batch_size=1000,
    hidden_dim = 200,
    preset = 'CIR_Feller_violated_high_gamma', # choices: ['GBM','CIR_Feller_satisfied','CIR_Feller_violated_moderate_gamma','CIR_Feller_violated_high_gamma'] 
    N_train = 100_000,
    N_test = 100_000,
    activation='leaky_relu',
    negative_slope=0.1, # Negative slope for leaky_relu activation 
    proc_type='scale_S_ref', # pre-processing type 
    output_activation=None, # output activation of the generator, discriminator activation currently fixed at sigmoid 
    device=DEVICE,
    beta1=0.5, # Adam beta_1
    beta2=0.999, # Adam beta_2 
    lr_G=0.0001, # base learning rate of the generator 
    lr_D=0.0005, # base learning rate of the discriminator 
    n_D = 2, # number of discriminator iterations per generator iteration, fixed to 1 for supervised GAN 
    supervised=False, # by default, use a vanilla GAN 
    seed=SEED,
    save_figs=False,
    save_iter_plot=True,
    report=False,
    results_path=DEFAULT_DIR,
    plot_interval = 500,
    )

torch.manual_seed(SEED)
np.random.seed(seed=SEED)


def train_GAN(netD,netG,data,meta,netD_Mil=None):  
    '''
    Training loop: train_GAN(netD,netG,data,meta)
    Inspired by several tricks from DCGAN PyTorch tutorial. 
    '''

    #-------------------------------------------------------
    # Initialisation 
    #-------------------------------------------------------    

    real_label = 1
    fake_label = 0    
    GANloss = nn.BCELoss() 

    GAN_step = step_handler(supervised_bool=meta['supervised'], CGAN_bool=data.CGAN)

    # Initialise lists for logging   
    D_losses = []
    G_losses = []
    times = []
    wasses = []
    ksstat = []
    delta_ts_passed = []
    D_grads = []
    G_grads = []

    # Short handle for training params 
    c_lr = meta['c_lr']
    cut_lr_every = meta['cut_lr_every']
    epochs = meta['epochs']
    results_path = meta['results_path']
    b_size = meta['batch_size']
    proc_type = meta['proc_type']
    device = meta['device']

    if not pt.exists(pt.join(results_path,'training','')):
        os.mkdir(pt.join(results_path,'training',''))
    train_analysis = CGANalysis(data,netD,netG,SDE=data.SDE,save_all_figs=meta['save_figs'],results_path=pt.join(meta['results_path'],'training',''),\
        proc_type=proc_type,eps=meta['eps'],supervised=meta['supervised'],device=meta['device'])
    train_analysis.format = 'png'

    if data.C is not None:
        C_tensors = dict_to_tensor(data.C)
        C_test = make_test_tensor(data.C_test,data.N_test)
    else:
        C_test = None 

    # Pre-allocate an empty array for each layer to store the norm 
    for l in range(count_layers(netD)):
            D_grads.append([])

    for l in range(count_layers(netG)):
            G_grads.append([])

    # Initialise counters 
    itervec = []
    iters = 0
    plot_iter = 0

    # Get the amount of batches implied by training set size and batch_size 
    n_batches = data.N//b_size

    optG = optim.Adam(netG.parameters(),lr=meta['lr_G'],betas=(meta['beta1'],meta['beta2']))
    optD = optim.Adam(netD.parameters(),lr=meta['lr_D'],betas=(meta['beta1'],meta['beta2']))

    #-------------------------------------------------------
    # Start training loop 
    #-------------------------------------------------------
    
    for epoch in range(epochs):
        tick0 = time.time()    
        for i in range(n_batches):
            if iters % cut_lr_every == 0:
                optG.param_groups[0]['lr'] = optG.param_groups[0]['lr']/c_lr

            # Sample random minibatch from training set with replacement
            indices = np.array((np.random.rand(b_size)*data.N),dtype=int)
            # Uncomment to sample minibatch from training set without replacement
            # indices = np.arange(i*b_size,(i+1)*b_size)

            # Get data batch based on indices 
            X_i = data.exact[indices,:].to(device)
            C_i = C_tensors[indices,:].to(device) if data.CGAN else None
            Z_i = data.Z[indices,:].to(device) if meta['supervised'] else None

            #-------------------------------------------------------
            # GAN training step 
            #-------------------------------------------------------

            errD,errG,D_x,D_G_z1,D_G_z2 = GAN_step(netD,netG,optD,optG,GANloss,device,X_i=X_i,C_i=C_i,Z_i=Z_i,real_label=real_label,fake_label=fake_label,n_D=meta['n_D'])

            #-------------------------------------------------------            

            # Output training stats
            if (iters % 100 == 0) and  (i % b_size) == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, data.N//b_size,
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                input_G = input_sample(data.N_test,C=C_test,device=device)
                fake = postprocess(netG(input_G).detach().view(-1),{**data.params,**data.C_test}['S0'],proc_type=proc_type,\
                    S_ref=torch.tensor(data.params['S_bar'],device=meta['device'],dtype=torch.float32),eps=meta['eps']).cpu().view(-1).numpy()
                ecdf_fake = smd.ECDF(fake)
                ecdf_test = smd.ECDF(data.exact_test.view(-1))
                # cdf_test = train_analysis.exact_cdf(params={**data.params,**C_test})
                x = np.linspace(1e-5,3,1000) # plotting vector 
                x = train_analysis.x

                # Infinity norm with ECDF on test data (Kolmogorov-Smirnov statistic)
                ksstat.append(np.max(np.abs(ecdf_fake(x)-ecdf_test(x))))
                # ksstat.append(stat.kstest(fake,cdf=cdf_test),alternative='two-sided')[0])

                # 1D Wasserstein distance as implemented in Scipy stats package
                wasses.append(stat.wasserstein_distance(fake,data.exact_test.view(-1)))

                # Keep track of the L1 norm of the gradients in each layer
                append_gradients(netD,netG,D_grads,G_grads)

                itervec.append(iters)

            # Update the generated data in analysis instance 
            if ((iters % meta['plot_interval'] == 0) and (meta['save_iter_plot'] == True)):
                # Update network references for inference 
                train_analysis.G = netG
                train_analysis.D = netD
                train_analysis.save_iter_plot(iters,params=data.params,D=netD,G=netG)
                plot_iter += 1

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        tick1 = time.time()
        times.append(tick1-tick0)

    #-------------------------------------------------------
    # Store training results 
    #-------------------------------------------------------    

    # Get range of training parameters if CGAN
    if data.C is not None:
        C_ranges = dict()
        for key in list(data.C.keys()):
            C_ranges[key] = (min(data.C[key]),max(data.C[key]))

    # Create dict for output log 
    output_dict = dict(
        iterations=np.arange(1,iters+1),
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
        train_condition = list(data.C.keys()) if data.C is not None else None,
        train_condition_ranges = [str(C_ranges)] if data.C is not None else None,
        test_condition=[str(data.C_test)] if data.C is not None else None,
        params_names = list(data.params),
        params=list(data.params.values()),
        SDE=[data.SDE],
        )

    # Add metaparameters to output log 
    output_dict = {**meta,**output_dict}

    for k in range(len(G_grads)):
        dict_entry = dict()
        dict_entry['G_grad_layer_%d'%k] = G_grads[k]
        output_dict.update(dict_entry)

    for k in range(len(D_grads)):
        dict_entry = dict()
        dict_entry['D_grad_layer_%d'%k] = D_grads[k]
        output_dict.update(dict_entry)

    # Convert to Pandas DataFrame 
    results_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in output_dict.items() ]))

    pd.concat((results_df,pd.DataFrame({**data.params,**data.C_test},index=[0])),axis=1,ignore_index=True)

    results_df = pd.concat([results_df,pd.DataFrame(G_grads[k],columns=['GradsG_L%d'%k])],axis=1,sort=False)
    for k in range(len(D_grads)):
        results_df = pd.concat([results_df,pd.DataFrame(D_grads[k],columns=['GradsD_L%d'%k])],axis=1,sort=False)

    print('----- Training complete -----')

    return output_dict, results_df

def main():
    # Supervised GAN? 
    options = [False,True]
    # Alternative: run over different pre-processing types, comment the above line and uncomment the one below
    # options = [None,'returns','logreturns','scale_S_ref']    

    results_path = META['results_path']    

    for i in range(len(options)):
        # Reset the seed at each iteration for equal initalisation of the nets 
        torch.manual_seed(SEED)
        np.random.seed(seed=SEED)
        META['seed'] = SEED

        # Make folder for each run of the training loop 
        if not pt.exists(pt.join(results_path,'iter_%d'%i)):
            os.mkdir(pt.join(results_path+'/iter_%d'%i))

        #---------------------------------------------------------
        # Modify training conditions in loop 
        #---------------------------------------------------------

        META['supervised'] = options[i]
        # Alternative: run over different pre-processing types, comment the above line and uncomment the one below 
        # META['proc_type'] = options[i]

        #---------------------------------------------------------

        # Override the default n_D, the amount of training steps of D per G training step if vanilla GAN. 
        META['n_D'] = 1 if META['supervised'] else META['n_D']

        #---------------------------------------------------------

        # Make the dataset and initialise the GAN 
        # X.generate_CIR_data()
        X = load_preset(META['preset'],N_train=META['N_train'],N_test=META['N_test'])
        X.exact = preprocess(X.exact,torch.tensor(X.params['S0'],dtype=torch.float32).view(-1,1),proc_type=META['proc_type'],\
        	S_ref=torch.tensor(X.params['S_bar'],device=torch.device('cpu'),dtype=torch.float32),eps=META['eps'])
        
        c_dim = 0 if X.C is None else len(X.C)
        netG = Generator(c_dim=c_dim).to(DEVICE)
        netG.eps=META['eps']
        netD = Discriminator(c_dim=c_dim+1,negative_slope=META['negative_slope'],hidden_dim=META['hidden_dim'],activation=META['activation']).to(DEVICE) if META['supervised']\
         else Discriminator(c_dim=c_dim,negative_slope=META['negative_slope'],hidden_dim=META['hidden_dim'],activation=META['activation']).to(DEVICE)
        analysis = CGANalysis(X,netD,netG,SDE=X.SDE,save_all_figs=META['save_figs'],results_path=results_path,proc_type=META['proc_type'],eps=META['eps'],supervised=META['supervised'])

        # Traing the GAN 
        output_dict, results_df = train_GAN(netD,netG,X,META)

        # Store results 
        netG_dir = pt.join(results_path,'iter_%d'%i,'netG.pth')
        netD_dir = pt.join(results_path,'iter_%d'%i,'netD.pth')
        torch.save(netG.state_dict(),netG_dir)
        print('Saved Generator in %s'%netG_dir)
        torch.save(netD.state_dict(),netD_dir)
        print('Saved Discriminator in %s'%netD_dir)

        if META['report'] == True:
                results_df.to_csv(pt.join(results_path,'iter_%d'%i, 'train_log.csv'),index=False,header=True)
                # Uncomment to save the entire output dict 
                # log_path = pt.join(results_path,'iter_%d'%i,'train_log.pkl')
                # pickle_it(output_dict,log_path)
                meta_path = pt.join(results_path,'iter_%d'%i,'metadata.pkl')
                pickle_it(META,meta_path)
                print('Saved logs in ' + results_path + '/iter_%d/'%i)

    print('----- Experiment finished -----')

if __name__=='__main__':
    main()