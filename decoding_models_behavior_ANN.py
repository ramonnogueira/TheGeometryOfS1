import os
import numpy as np
import scipy
import math
import sys
import torch
import pandas
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import nn_pytorch_encoders
import features_whiskers
import spikes_processing
import licking_preprocessing
import decoders_classes
import functions_miscellaneous
nan=float('nan')

# Parameters
t_steps=30
t_dec=20  
stim_nat='Poisson'
dim_in=3
n_hidden=60
t_rnd_delay=10
sigma_test=1
n_test=40
n_files=5

pathload_pre='/home/ramon/Dropbox/chris_randy/data/data_ANN/'
path_save='/home/ramon/Dropbox/chris_randy/data_behavior/data_ANN/'
   
task_vec=['xor']#,'2d_xor','xor' 
model_vec=['FC_trivial','FC_one_deep_hundred_wide','FC_two_deep_hundred_wide','FC_three_deep_hundred_wide']  
reg_vec=np.logspace(-8,2,10)                                                                                                                                                               
lr_vec=np.logspace(-6,0,10)
n_val=4
n_cv=4

for i in range(len(task_vec)):
    print ('Task %s'%task_vec[i])
    for hh in range(n_files):
        datos=open(pathload_pre+'data_recurrent_task_%s_%i_%i_steps_%s_n_hidden_%i_dim_in_%i_t_delay_%i_noise_%.1f_n_test_%i_file_%i.pkl'%(task_vec[i],t_steps,t_dec,stim_nat,n_hidden,dim_in,t_rnd_delay,sigma_test,n_test,hh),'rb')
        data=pkl.load(datos)
        datos.close()
        for ii in range(len(model_vec)):   
            print (model_vec[ii])
            for iii in range(len(reg_vec)):
                print ('  reg ',reg_vec[iii])
                feat_pre=data['input_rec_test'][:,0:t_dec] # Only the time used by the network
                feat_shape=np.shape(feat_pre)
                feat_norm=np.zeros((feat_shape[0],1,feat_shape[2],feat_shape[1])) # We change the shape to Batch size x features x timesteps
                for k in range(feat_shape[0]):
                    feat_norm[k,0]=np.transpose(feat_pre[k])

                dec_cl_nn=decoders_classes.feedforward_validation(feat=feat_norm,clase=data['class_test'],reg=reg_vec[iii],lr_vec=lr_vec,n_val=n_val,n_cv=n_cv)
                dec_nn=dec_cl_nn.logregress_ANN(model=model_vec[ii])
                torch.save(dec_nn['models'],path_save+'models_decoders_%s_%s_%i_hidden_%i_%i_noise_%.1f_file_%i.pt'%(task_vec[i],model_vec[ii],iii,n_hidden,t_rnd_delay,sigma_test,hh))
                pkl.dump({'performance':dec_nn['performance']},open(path_save+'performance_decoders_%s_%s_%i_hidden_%i_delay_%i_noise_%.1f_file_%i.pkl'%(task_vec[i],model_vec[ii],iii,n_hidden,t_rnd_delay,sigma_test,hh),'wb'))


      
      
