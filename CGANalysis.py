# Jorino van Rhijn
# Monte Carlo Simulation of SDEs with GANs

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.distributions as smd
import scipy.stats as stat
from utils import input_sample, count_layers, make_test_tensor, postprocess, preprocess
from KDEpy import FFTKDE
from matplotlib.legend_handler import HandlerTuple


class CGANalysis():
    def __init__(self, data, D, G, SDE='CIR', method='iid_Z', save_all_figs=False, results_path=None, proc_type=None,
                 plotstyle='default', eps=1e-8, supervised=False, device=None):

        # Create a grid on the problem domain for plotting
        a, b = self.get_plot_bounds(data.exact.numpy())
        if SDE == 'GBM':
            a = 0
        elif SDE == 'CIR':
            if a <= 1e-6:
                a = eps

        if device is None:
            device = 'cpu'
        self.device = device

        # Default parameters
        plt.style.use(plotstyle)
        self.x = np.linspace(a, b, 1000)
        self.N = data.N
        self.N_test = data.N_test
        self.C_test = data.C_test.copy()
        self.SDE = SDE
        self.method = method
        self.proc_type = proc_type
        self.save_all_figs = save_all_figs
        self.results_path = results_path
        self.fixed_noise = torch.randn((self.N_test, 1), device=self.device)
        self.CGAN = data.CGAN
        self.eps = eps

        # Save pointers to networks
        self.D = D
        self.G = G

        # Copy dataset parameters, as the class writes to them
        self.params = data.params.copy()

        # Case discriminator informed with Z
        self.supervised = supervised

        # Set default file extension to pdf
        self.format = 'pdf'

        # Make the exact pdf, cdf and sampling scheme available as methods
        if self.SDE == 'GBM':
            self.exact_cdf = data.exact_cdf_GBM
            self.exact_pdf = data.exact_pdf_GBM
            self.sample_exact = data.get_exact_GBM_samples
        elif self.SDE == 'CIR':
            self.exact_cdf = data.exact_cdf_CIR
            self.exact_pdf = data.exact_pdf_CIR
            self.sample_exact = data.get_exact_CIR_samples
        else:
            raise Exception(f'Type of dynamics not understood: {self.SDE}')

    @staticmethod
    def get_plot_bounds(X):
        '''
        Get plotting domain based on a sample of data. Alternative to autoscaling
        Input: numpy array X with samples from distribution of interest
        '''
        a = X.min()-0.1*np.abs(X.min())
        b = X.max()+0.1*np.abs(X.max())
        a = np.min((a, b))
        b = np.max((a, b))
        if a == b:
            a -= 1e-20  # Prevent situation a=b if both are 0
        return a, b

    def get_exact_pdf(self, params):
        '''
        Get bound method of pdf corresponding to self.SDE
        '''
        if self.SDE == 'GBM':
            return self.exact_pdf(params)
        elif self.SDE == 'CIR':
            return self.exact_pdf(params)
        else:
            raise Exception(f'SDE type \'{self.SDE}\' not supported or not understood.')

    def get_exact_cdf(self, params):
        '''
        Get bound method of cdf corresponding to self.SDE
        '''
        if self.SDE == 'GBM':
            return self.exact_cdf(params)
        elif self.SDE == 'CIR':
            return self.exact_cdf(params)
        else:
            raise Exception(f'SDE type \'{self.SDE}\' not supported or not understood.')

    def ecdf_plot(self, C=None, G=None, params=None, save=False, filename=None, raw_output=False, save_format=None, x_plot=None, legendname=None, grid=False):
        '''
        Plotting function that creates [ECDF plot]. 
        '''
        if (not save) and self.save_all_figs:
            save = True
        if params is None:
            params = self.params.copy()
        if G is None:
            G = self.G
        if self.CGAN:
            params = {**params, **self.C_test}
            assert self.CGAN and (G.c_dim > 0), 'Generator appears not to be trained as a CGAN, while CGAN is toggled.'
            assert len(C) == 1, 'Only one plot condition is supported. To fix other parameters, specify them in self.C_test'
        if save_format is None:
            save_format = self.format
        if save:
            assert filename is not None, 'Please specify a filename to save the figure to.'
            
        # Define plotting linestyles 
        lines = ['dashed','solid','dotted','dashdot']
        # Initialise handles for legend entries 
        handles_exact = []
        handles = []
        names = []
        fig,ax = plt.subplots(1,1,dpi=100)
        
        if G.c_dim > 0:
            #------------------------------------------------
            # Case 1: Conditional GAN 
            #------------------------------------------------
            # Get name of the conditional parameter
            c_name = next(iter(C.keys()))
            # Prepare the base strong of legend name 
            if legendname is None:
                legendname = c_name            
            # Get array with plot values
            cs = np.array(next(iter(C.values())))

            for i,c in enumerate(cs):
                # Cycle between linestyles
                line = lines[i%len(lines)]
                # First cast the current condition back to a dict
                c_dict = dict()
                c_dict[c_name] = c
                # Cast current condition into tensor, replacing the relevant value in C_test
                c_tensor = make_test_tensor({**self.C_test,**c_dict},self.N_test,device=self.device)
                # Get an input sample 
                in_sample = input_sample(self.N_test,C=c_tensor,device=self.device)
                # Infer with generator
                output = G(in_sample).detach()
                gendata = postprocess(output,{**params,**c_dict}['S0'],proc_type=self.proc_type,delta_t=torch.tensor(params['t']),S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).view(-1).cpu().numpy()

                if raw_output and (self.proc_type is not None):
                    if x_plot is None:
                        a,b = self.get_plot_bounds(output)
                        x = np.linspace(a,b,1000)
                    else:
                        x = x_plot    
                    # Option to plot output before postprocessing 
                    ecdf_gen_data = smd.ECDF(output.view(-1).cpu().numpy())
                    # Get pre-processed exact variates for estimate of exact cdf 
                    exact_raw = preprocess(self.sample_exact(N=self.N_test,params={**params,**c_dict}),torch.tensor({**params,**c_dict}['S0'],\
                        dtype=torch.float32).view(-1,1),proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).view(-1).cpu().numpy()
                    exact_cdf = smd.ECDF(exact_raw)

                    l_e, = ax.plot(x,exact_cdf(x),'k',linestyle=line)
                    l_g, = ax.plot(x,ecdf_gen_data(x),'-')
                    handles_exact.append(l_e)
                    handles.append(l_g)
                    names.append(legendname+f'= {c}')
                    ax.set_xlabel('$R_t$')
                else:
                    if x_plot is None:
                        a = 0.1*gendata.min()
                        b = 1.1*gendata.max()
                        if a <= 1e-4:
                            a = self.eps
                        x = np.linspace(a,b,1000)
                    else:
                        x = x_plot                    

                    # Plot output after pre-processing 
                    ecdf_gen_data = smd.ECDF(gendata)
                    # Instantiate function for cdf of analytical distribution
                    exact_cdf = self.get_exact_cdf({**params,**c_dict})

                    l_e, = ax.plot(x,exact_cdf(x),'k',linestyle=line)
                    l_g, = ax.plot(x,ecdf_gen_data(x),'-')
                    handles_exact.append(l_e)
                    handles.append(l_g)
                    names.append(legendname+f'={c}')
                    ax.set_xlabel('$S_{t+\Delta t} \mid S_t$')

            # Make final handles for legend            
            names.append('Exact')         
            handles.append(tuple(handles_exact))
            ax.legend(handles, names, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

        else:
            #------------------------------------------------
            # Case 2: Unconditional GAN 
            #------------------------------------------------    
            in_sample = input_sample(self.N_test,device=self.device)
            # Infer with generator
            output = G(in_sample).detach()
            gendata = postprocess(output,params['S0'],proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).view(-1).cpu().numpy()
            
            if raw_output and (self.proc_type is not None):
                if x_plot is None:
                    a,b = self.get_plot_bounds(output)
                    x = np.linspace(a,b,1000)
                else:
                    x = x_plot
                ecdf_gen_data = smd.ECDF(output.view(-1).cpu().numpy())
                exact_raw = preprocess(self.sample_exact(N=self.N_test,params=params),torch.tensor(params['S0'],\
                        dtype=torch.float32).view(-1,1),proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).view(-1).cpu().numpy()
                ax.plot(x,smd.ECDF(exact_raw)(x),'Exact')
                ax.plot(x,smd.ECDF(output.view(-1).cpu().numpy())(x),'-',label='Generated')
                ax.set_xlabel('$R_t$')
            else:    
                a = 0.1*gendata.min()
                b = 1.1*gendata.max()
                if a <= 1e-4:
                    a = self.eps
                x = np.linspace(a,b,1000)
                ecdf_gen_data = smd.ECDF(gendata)
                # Instantiate functions for cdf and pdf of analytical distribution
                exact_cdf = self.get_exact_cdf(params)
                ax.plot(x,exact_cdf(x),'--k',label='Exact')
                ax.plot(x,ecdf_gen_data(x),label='Generated')
                ax.set_xlabel('$S_{t+\Delta t} \mid S_t$')

            # Optional manual setting of horizontal axis limits
            if x_lims is not None:
                ax.set_xlim(x_lims)
            
            ax.legend()

        # Uncommented for paper 
        # fig.suptitle(c_name + f' = {cs}')
        # fig.suptitle(f'{str( {**self.C_test,**C}) }')

        #------------------------------------------------
        # Wrap up 
        #------------------------------------------------   
        
        if grid:
            plt.grid('on')

        if (save == True):
            plt.savefig(filename,format=self.format)
            print(f'Saved ecdf_plot in folder at {filename}')
            plt.close()
        else:
            plt.show()  

    def kde_plot(self,C=None,G=None,params=None,save=False,filename=None,raw_output=False,save_format=None,x_lims=None):
        '''
        Plotting function that creates [kde plot]. 
        '''

        if params is None:
            params = self.params.copy()
        if G is None:
            G = self.G
        if self.CGAN:
            params = {**params,**self.C_test}
            assert self.CGAN and (G.c_dim > 0), 'Generator appears not to be trained as a CGAN, while CGAN is toggled.'
            assert len(C) == 1, 'Only one plot condition is supported. To fix other parameters, specify them in self.C_test'
        if save_format is None:
            save_format = self.format
        if save:
            assert filename is not None, 'Please specify a filename to save the figure to.'
            
        # Define plotting linestyles 
        lines = ['dashed','solid','dotted','dashdot']
        # Initialise handles for legend entries 
        handles_exact = []
        handles = []
        names = []
        fig,ax = plt.subplots(1,1,dpi=100)
        
        if G.c_dim > 0:
            #------------------------------------------------
            # Case 1: Conditional GAN 
            #------------------------------------------------
            # Get name of conditional parameter
            c_name = next(iter(C.keys()))
            # Get array with plot values
            cs = np.array(next(iter(C.values())))

            for i,c in enumerate(cs):
                # Cycle between linestyles
                line = lines[i%len(lines)]
                # First cast the current condition back to a dict
                c_dict = dict()
                c_dict[c_name] = c
                # Cast current condition into tensor, replacing the relevant value in C_test
                c_tensor = make_test_tensor({**self.C_test,**c_dict},self.N_test,device=self.device)
                # Get an input sample 
                in_sample = input_sample(self.N_test,C=c_tensor,device=self.device)
                # Infer with generator
                output = G(in_sample).detach()
                gendata = postprocess(output,{**params,**c_dict}['S0'],proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).cpu().view(-1).numpy()


                if raw_output and (self.proc_type is not None):
                    # Get pre-processed exact variates for estimate of exact pdf
                    exact_raw = preprocess(self.sample_exact(N=self.N_test,params={**params,**c_dict}),torch.tensor({**params,**c_dict}['S0'],\
                        dtype=torch.float32).view(-1,1),proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).cpu().view(-1).numpy()

                    x_out_kde_exact, kde_exact = FFTKDE(kernel='gaussian',bw='silverman').fit(exact_raw)(self.N_test)
                    x_out_kde_gen, kde_gen = FFTKDE(kernel='gaussian',bw='silverman').fit(output.view(-1).cpu().numpy())(self.N_test)

                    l_e, = ax.plot(x_out_kde_exact,kde_exact,'k',linestyle=line)
                    l_g, = ax.plot(x_out_kde_gen,kde_gen,'-')

                    # Define the color if only one condition is chosen 
                    if (len(cs) == 1):
                        l_g.set_color('darkorange')

                    ax.fill_between(x_out_kde_gen, y1=kde_gen, alpha=0.25, facecolor=l_g.get_color())
                    plt.autoscale(enable=True, axis='x', tight=True)
                    ax.set_ylim(bottom=0)
                    handles_exact.append(l_e)
                    handles.append(l_g)
                    names.append(c_name+f' = {c}')
                    ax.set_xlabel('$R_t$')
                else:
                    a = 0.1*gendata.min()
                    b = 1.1*gendata.max()
                    if a <= 1e-4:
                        a = self.eps
                    x = np.linspace(a,b,1000)
                    # Instantiate function for pdf of analytical distribution
                    exact_pdf = self.get_exact_pdf({**params,**c_dict})
                    x_out_kde_gen, kde_gen = FFTKDE(kernel='gaussian',bw='silverman').fit(gendata)(self.N_test)

                    l_e, = ax.plot(x,exact_pdf(x),'k',linestyle=line)
                    l_g, = ax.plot(x_out_kde_gen,kde_gen,'-')

                    # Define the color if only one condition is chosen 
                    if (len(cs) == 1):
                        l_g.set_color('darkorange')
                    ax.fill_between(x_out_kde_gen, y1=kde_gen, alpha=0.25, facecolor=l_g.get_color())
                    handles_exact.append(l_e)
                    handles.append(l_g)
                    names.append(c_name+f' = {c}')
                    ax.set_xlabel('$S_t$')

            # Make final handles for legend            
            names.append('Exact')         
            handles.append(tuple(handles_exact))
            ax.legend(handles, names, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

            # Optional manual setting of horizontal axis limits
            if x_lims is not None:
                ax.set_xlim(x_lims)
            else:
                ax.autoscale(enable=True, axis='x', tight=True)
                ax.autoscale(enable=True, axis='y')
                ax.set_ylim(bottom=0)

        else:
            #------------------------------------------------
            # Case 2: Unconditional GAN 
            #------------------------------------------------    
            in_sample = input_sample(self.N_test,device=self.device)
            # Infer with generator
            output = G(in_sample).detach()
            gendata = postprocess(output,params['S0'],proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).cpu().view(-1).numpy()
            
            if raw_output and (self.proc_type is not None):
                exact_raw = preprocess(self.sample_exact(N=self.N_test,params=params),torch.tensor(params['S0'],\
                        dtype=torch.float32).view(-1,1),proc_type=self.proc_type,K=calc_K(params,proc_type=self.proc_type,SDE=self.SDE),S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).cpu().view(-1).numpy()
                sns.kdeplot(exact_raw,label='Exact',ax=ax,linestyle='--',color='k')
                sns.kdeplot(output,label='Generated',shade=True,color='darkorange')
                ax.set_xlabel('$R_t$')
            else:    
                a = 0.1*gendata.min()
                b = 1.1*gendata.max()
                if a <= 1e-4:
                    a = self.eps
                x = np.linspace(a,b,1000)
                ecdf_gen_data = smd.ECDF(gendata)
                # Instantiate functions for cdf and pdf of analytical distribution
                exact_pdf = self.get_exact_pdf(params)
                ax.plot(x,exact_pdf(x),'--k',label='Exact')
                sns.kdeplot(gendata,label='Generated',ax=ax,shade=True,color='DarkOrange')
                ax.set_xlabel('$S_t$')
            
            ax.legend()
            # Optional manual setting of horizontal axis limits
            if x_lims is not None:
                ax.set_xlim(x_lims)

        # fig.suptitle(c_name + f' = {cs}')
        # fig.suptitle(f'{str( {**self.C_test,**C}) }')

        #------------------------------------------------
        # Wrap up 
        #------------------------------------------------   
        
        if (save == True):
            plt.savefig(filename,format=self.format)
            print(f'Saved kde_plot in folder at {filename}')
            plt.close()
        else:
            plt.show()  
        
    def save_iter_plot(self,iteration,filename=None,C=None,G=None,D=None,params=None,save_conf=True):
        '''
        Plots a kde plot and the confidence of the discriminator D(x). Saves the plot in `filename'. 
        '''
        if params is None:
            params = self.params.copy()
        if G is None:
            G = self.G
        if D is None:
            D = self.D
        if (C is None) and self.CGAN:
            C = self.C_test
        if filename is None:
            filename = os.path.join(self.results_path,'plot_iter_%02d.'%iteration + self.format)
        if self.CGAN:
            assert G.c_dim > 0, 'Generator must be a conditional GAN if CGAN is toggled in the dataset.'
            assert str(C.keys()) == str(self.C_test.keys()), 'The tensor specified must include all conditional parameters, in the same order as C_test.'
        else:
            # Vanilla GAN case
            C = dict()

        # Update the relevant parameters with C
        params = {**params,**C}

        #------------------------------------------------------
        # Compute inputs
        #------------------------------------------------------

        if self.CGAN:
            C_test_tensor = make_test_tensor(C,self.N_test,device=self.device)
            in_sample = input_sample(self.N_test,C=C_test_tensor,Z=self.fixed_noise.to(self.device),device=self.device)
        else:
            in_sample = input_sample(self.N_test,C=None,Z=self.fixed_noise,device=self.device)

        output = G(in_sample).detach()
        gendata = postprocess(output.view(-1),params['S0'],proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).cpu().view(-1).numpy()

        # Instantiate function for pdf of analytical distribution, add 1e-6 to keep the fraction X_next/X_prev finite
        exact_raw = preprocess(self.sample_exact(N=self.N_test,params=params),torch.tensor(params['S0'],\
                        dtype=torch.float32).view(-1,1),proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=torch.device('cpu'),dtype=torch.float32),eps=self.eps).cpu().view(-1).numpy()
        
        output = output.view(-1).cpu().numpy()
        
        # Define domain based on GAN output 
        a1 = np.min(output)-0.1*np.abs(output.min())
        b1 = np.min(exact_raw)-0.1*np.abs(exact_raw.min())
        a2 = np.max(output)+0.1*np.abs(output.max())
        b2 = np.max(exact_raw)+0.1*np.abs(exact_raw.max())
        a = np.min((a1,b1))
        b = np.max((a2,b2))
        if a == 0:
            a -= 1e-20        

        if not self.supervised:

            # Define grid for KDE to be computed on 
            x_opt = np.linspace(a,b,1000)

            # Compute exact density p* and generator density p_th

            if self.proc_type is None:
                # Use exact pdf for p* if no pre-processing is used
                p_star = self.get_exact_pdf(params)(x_opt)
            else:
                # Otherwise use kernel estimate to compute p*
                p_star = FFTKDE(kernel='gaussian',bw='silverman').fit(exact_raw).evaluate(x_opt)

            kde_th = FFTKDE(kernel='gaussian',bw='silverman').fit(output)
            p_th = kde_th.evaluate(x_opt)

            # Optimal discriminator given G
            D_opt = p_star/(p_th+p_star)

            x_D = torch.linspace(x_opt.min(),x_opt.max(),self.N_test)

            # Build the input to the discriminator 
            input_D = x_D.view(-1,1).to(self.device)
            if self.supervised:
                # If the discriminator is informed with Z, give it zeros for testing 
                input_D = torch.cat((input_D,torch.zeros(self.N_test).view(-1,1)),axis=1)
            if self.CGAN:
                input_D = torch.cat((input_D,C_test_tensor),axis=1)
   
   
        #------------------------------------------------------
        # Select amount of subplots to be shown 
        #------------------------------------------------------
        # Only plot pre-processed data if pre-processing is not None
        # Only plot discriminator confidence if vanilla GAN is used

        single_handle = False # toggle to use if the axis handle is not an array
        if (self.proc_type is None) and (self.supervised):
            fig,ax = plt.subplots(1,1,figsize=(10,10),dpi=100)
            title_string = 'Generator output'
            single_handle = True
        elif (self.proc_type is None) and (not self.supervised):
            fig,ax = plt.subplots(1,2,figsize=(20,10),dpi=100)
            title_string = 'Generator output'
        elif (self.proc_type is not None) and (self.supervised):
            fig,ax = plt.subplots(1,2,figsize=(20,10),dpi=100)
            title_string = 'Post-processed data'
        else:
            fig,ax = plt.subplots(1,3,figsize=(30,10),dpi=100)
            title_string = 'Post-processed data'

        k_ax = 0 # counter for axis index 

        #------------------------------------------------------
        # Plot 1: Post-processed data
        #------------------------------------------------------
        y = self.x
        ymin = y.min()-0.1*np.abs(y.min())
        ymax = y.max()+0.1*np.abs(y.max())

        exact_pdf = self.get_exact_pdf(params)            

        if single_handle:
            ax_plot_1 = ax
        else:
            ax_plot_1 = ax[k_ax]

        ax_plot_1.plot(y,exact_pdf(y),'--k',label='Exact pdf')
        sns.kdeplot(gendata,shade=True,ax=ax_plot_1,label='Generated data')
        ax_plot_1.set_xlabel('$S_t$')
        # fig.suptitle(f'time = {self.T}')
        ax_plot_1.legend()
        # ax_plot_1.set_xlim(xmin=ymin,xmax=ymax)
        ax_plot_1.autoscale(enable=True, axis='x', tight=True)
        ax_plot_1.autoscale(enable=True, axis='y')
        ax_plot_1.set_ylim(bottom=0)        
        ax_plot_1.set_title(title_string)

        # Also plot only the kde plot as pdf 
        f_kde,ax_kde = plt.subplots(1,1,dpi=100)
        ax_kde.plot(y,exact_pdf(y),'--k',label='Exact pdf')
        sns.kdeplot(gendata,shade=True,ax=ax_kde,label='Generated data')
        ax_kde.set_xlabel('$S_t$')
        ax_kde.legend()
        ax_kde.set_xlim(xmin=ymin,xmax=ymax)
        # ax_kde.set_title(title_string)
        f_kde.suptitle(f'Iteration {iteration}')
        f_kde.savefig(os.path.join(self.results_path,'kde_output_iter_%02d'%iteration+'.pdf'),format='pdf')
        plt.close(f_kde)

        #------------------------------------------------------
        # Plot 2: Generator output
        #------------------------------------------------------
        if self.proc_type is not None:
            k_ax += 1
            sns.kdeplot(exact_raw,linestyle='--',color='k',ax=ax[k_ax],label='Pre-processed exact')        
            sns.kdeplot(output,shade=True,ax=ax[k_ax],label='Generated data')
            ax[k_ax].set_xlabel('$R_t$')
            ax[k_ax].legend()
            # ax[k_ax].set_xlim(xmin=a,xmax=b)
            ax[k_ax].autoscale(enable=True, axis='x', tight=True)
            ax[k_ax].autoscale(enable=True, axis='y')
            ax[k_ax].set_ylim(bottom=0)            
            ax[k_ax].set_title('Generator output')

        #------------------------------------------------------
        # Plot 3: Discriminator confidence
        #------------------------------------------------------    

        if not self.supervised:
            k_ax += 1
            ax[k_ax].plot(x_D,D(input_D).view(-1,1).detach().view(-1).cpu().numpy(),label='Discriminator output')
            ax[k_ax].plot(x_opt,D_opt,'--k',label='Optimal discriminator')

            # ax[1].set_title('Discriminator confidence')
            if self.proc_type is None:
                ax[k_ax].set_xlabel('$S_t$')
            else:
                ax[k_ax].set_xlabel('$R_t$')
            ax[k_ax].legend()
            # ax[k_ax].set_xlim(xmin=a,xmax=b)

            ax[k_ax].autoscale(enable=True, axis='x', tight=True)
            ax[k_ax].autoscale(enable=True, axis='y')
            ax[k_ax].set_ylim(bottom=0)

            if save_conf:
            # Repeat plot to save discriminator confidence itself as well 
                f_conf,ax_conf = plt.subplots(1,1,dpi=100)
                ax_conf.plot(x_D,D(input_D).view(-1,1).detach().view(-1).cpu().numpy(),label='Discriminator output')
                ax_conf.plot(x_opt,D_opt,'--k',label='Optimal discriminator')

                if self.proc_type is None:
                    ax_conf.set_xlabel('$S_t$')
                else:
                    ax_conf.set_xlabel('$R_t$')
                ax_conf.legend()
                ax_conf.set_xlim(xmin=a,xmax=b)
                f_conf.suptitle(f'Iteration {iteration}')
                f_conf.savefig(os.path.join(self.results_path,'D_conf_iter_%02d'%iteration+'.pdf'),format='pdf')
                plt.close(f_conf)

        #------------------------------------------------------
        # Wrap up
        #------------------------------------------------------

        fig.suptitle(f'Iteration {iteration}')
        fig.savefig(filename,format=self.format)
        plt.close()


    def QQ_plots(self,C=None,n_plot=None,G=None,params=None,results_path=None,save=False):
        '''
        Method to create QQ plots of generated data, test data and train data. 
        '''
        if (not save) and self.save_all_figs:
            save = True
        if params is None:
            params = self.params
        if results_path is None:
            results_path = self.results_path
        if G is None:
            G = self.G
        if (C is None) and self.CGAN:
            C = self.C_test
            assert G.c_dim > 0, 'Generator must be a conditional GAN if CGAN is toggled in the dataset.'
        elif self.CGAN:
            assert G.c_dim > 0, 'Generator must be a conditional GAN if CGAN is toggled in the dataset.'
            assert str(C.keys()) == str(self.C_test.keys()), 'The tensor specified must include all conditional parameters, in the same order as C_test.'
        else:
            # Vanilla GAN case
            C = dict()    
        if n_plot is None:
            if self.format == 'pdf':
                n_plot = 1000
            else:
                n_plot = self.N_test
        # Update parameters accordingly. If vanilla GAN, params remain unchanged 
        params = {**params,**C}

        if self.CGAN:
            C_test_tensor = make_test_tensor(C,n_plot,device=self.device)
            in_sample = input_sample(n_plot,C=C_test_tensor,device=self.device)
        else:
            in_sample = input_sample(n_plot,C=None,device=self.device)

        gendata = postprocess(G(in_sample).detach(),params['S0'],proc_type=self.proc_type,S_ref=torch.tensor(params['S_bar'],device=self.device,dtype=torch.float32),eps=self.eps).cpu().view(-1).numpy()

        if self.SDE == 'GBM':
            mu = params['mu']
            sigma = params['sigma']
            S0 = params['S0']
            t = params['t']

            scale = np.exp(np.log(S0)+(mu-0.5*sigma**2)*(t))
            s = sigma*np.sqrt(t)

            dist = stat.lognorm
            sparams = (s,0,scale)
        elif self.SDE == 'CIR':
            kappa = params['kappa']
            gamma = params['gamma']
            S_bar = params['S_bar']
            S0 = params['S0']
            s = params['s']
            t = params['t']
                
            kappa_bar = (4*kappa*S0*np.exp(-kappa*(t-s)))/(gamma**2*(1-np.exp(-kappa*(t-s))))
            c_bar = (gamma**2)/(4*kappa)*(1-np.exp(-kappa*(t-s)))    
            delta = (4*kappa*S_bar)/(gamma**2)

            dist = stat.ncx2
            sparams = (delta,kappa_bar,0,c_bar)
        else:
            raise Exception('SDE type not supported or understood.')

        # Test set 
        plt.figure(dpi=100)
        stat.probplot(x=self.sample_exact(N=n_plot,params=params).view(-1).cpu().numpy(),dist=dist,sparams=sparams,plot=plt)
        plt.title('')        
        if save:
            plt.savefig(results_path+'_QQ_test.'+self.format,format=self.format)
        else:
            plt.show()
        plt.close()

        # Generated
        plt.figure(dpi=100)
        stat.probplot(x=gendata,dist=dist,sparams=sparams,plot=plt)
        plt.title('')
        if save:
            plt.savefig(results_path+'_QQ_generated.'+self.format,format=self.format)
        else:
            plt.show()
        plt.close()

        if save:
            print('Saved QQ plot of exact variates and generated data in folder %s'%results_path)
        
    def IPM_plot(self,results):
        itervec = results['iters']
        KS_stats = results['KS_stat_test']
        wass_dists = results['Wass_dist_test']
        plt.figure()
        plt.plot(results['iters'],KS_stats,label='KS statistic with test data')
        plt.plot(results['iters'],wass_dists,label='1-Wass dist with test data')
        plt.xlabel('iteration')
        plt.legend()
        if self.save_all_figs:
            plt.savefig(self.results_path+'/_IPM_plot.png')
        
    def plot_grads(self,results,dest=None,save=None):
        '''
        Plot and optionally save the L1 norm of the discriminator and generator gradients.  
        Automatically saves figures if the class `.save_all_figs` toggle is set to True 

        --------
        Input: \n
        results, Pandas DataFrame output of training loop \n
        optional: 
        dest, strong with base path. Defaults to analysis class default path. \n
        save, set to True if this figure is the be saved, even if save_all_figs is set to False 
        --------
        '''

        if save is None:
            save=self.save_all_figs

        # Generator 
        L_G = count_layers(self.G)
        fig,ax = plt.subplots(1,1,dpi=100) 
        [ax.plot(results['iters'],results['G_grad_layer_%d'%k],label='layer %d'%k) for k in range(L_G)]
        ax.set_xlabel('iterations')
        ax.set_title('Generator')
        ax.legend()

        if self.save_all_figs or save:
            if dest:
                plt.savefig(dest+'/_G_grads.%s'%self.format,format=self.format)
            else:
                plt.savefig(self.results_path+'/_G_grads.%s'%self.format,format=self.format)
            plt.close()
        else:
            plt.show()
        
        # Discriminator 
        fig,ax = plt.subplots(1,1,dpi=100) 
        L_D = count_layers(self.D)
        [ax.plot(results['iters'],results['D_grad_layer_%d'%k],label='layer %d'%k) for k in range(L_D)]
        ax.set_xlabel('iterations')
        ax.set_title('Disciminator')
        ax.legend()
        
        if self.save_all_figs or save:
            if dest:
                plt.savefig(dest+'/_D_grads.%s'%self.format,format=self.format)
            else:
                plt.savefig(self.results_path+'/_D_grads.%s'%self.format,format=self.format)
            plt.close()
        else:
            plt.show()
