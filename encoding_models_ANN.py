import os
import matplotlib.pylab as plt
import numpy as np
import scipy
import math
import sys
import tables
import pandas
import torch
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from scipy.stats import binom
from scipy.stats import sem
nan=float('nan')
minf=float('-inf')
pinf=float('inf')

import functions_miscellaneous
import encoders_classes

# Parameters
t_steps=30
t_dec=20     
stim_nat='Poisson'
dim_in=3
n_hidden=60
t_rnd_delay=10
sigma_test=1.0
size_kernel=5
n_test=40
n_files=5
loss_type='gaussian'

pathload_pre='/home/ramon/Dropbox/chris_randy/data/data_ANN/'
path_save='/home/ramon/Dropbox/chris_randy/data_analysis/data_ANN/'

task_vec=['2d_xor']#,'xor']
model_vec=['FC_trivial','FC_one_deep_hundred_wide','FC_two_deep_hundred_wide','FC_three_deep_hundred_wide']
reg_vec=np.logspace(-8,2,10) 
lr_vec=np.logspace(-6,0,10)
n_val=4
n_cv=4

for i in range(len(task_vec)):
    print ('Task %s'%task_vec[i])
    for hh in range(3,n_files):
        print (hh)
        datos=open(pathload_pre+'data_recurrent_task_%s_%i_%i_steps_%s_n_hidden_%i_dim_in_%i_t_delay_%i_noise_%.1f_n_test_%i_file_%i.pkl'%(task_vec[i],t_steps,t_dec,stim_nat,n_hidden,dim_in,t_rnd_delay,sigma_test,n_test,hh),'rb')
        data=pkl.load(datos)
        datos.close()
        for ii in range(len(model_vec)):   
            print (model_vec[ii])
            for iii in range(len(reg_vec)):
                print ('  reg ',reg_vec[iii])
                enc_cl_nn=encoders_classes.feedforward_validation_ANN(feat=data['input_rec_test'],target=data['firing_rate_test'],lr_vec=lr_vec,reg=reg_vec[iii],loss_type=loss_type,n_val=n_val,n_cv=n_cv)
                enc_nn=enc_cl_nn.validation(model=model_vec[ii],size_kernel=size_kernel)
                torch.save(enc_nn['models'],path_save+'models_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f_n_test_%i_file_%i.pt'%(loss_type,task_vec[i],model_vec[ii],iii,size_kernel,n_hidden,t_rnd_delay,sigma_test,n_test,hh))
                pkl.dump({'performance':enc_nn['performance']},open(path_save+'performance_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f_n_test_%i_file_%i.pkl'%(loss_type,task_vec[i],model_vec[ii],iii,size_kernel,n_hidden,t_rnd_delay,sigma_test,n_test,hh),'wb'))
                pkl.dump({'index':enc_nn['index']},open(path_save+'index_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f_n_test_%i_file_%i.pkl'%(loss_type,task_vec[i],model_vec[ii],iii,size_kernel,n_hidden,t_rnd_delay,sigma_test,n_test,hh),'wb'))

        
