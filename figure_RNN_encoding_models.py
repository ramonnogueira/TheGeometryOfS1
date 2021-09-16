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
import matplotlib.pyplot as plt

import nn_pytorch_encoders
import features_whiskers
import spikes_processing
import licking_preprocessing
import decoders_classes
import functions_miscellaneous
nan=float('nan')
minf=float('-inf')
pinf=float('inf')

# Parameters
t_steps=30
t_dec=20  
stim_nat='Poisson'
dim_in=3
n_hidden=60
t_rnd_delay=10
sigma_noise=1.0
n_test=40
loss_type='gaussian'
size_kernel=5
files_vec=[0,1,2,4] 

path_load='/home/ramon/Dropbox/chris_randy/data_analysis/data_ANN/'
path_save='/home/ramon/Dropbox/chris_randy/plots'
   
task_vec=['linear']
model_vec=['FC_trivial','FC_one_deep_hundred_wide','FC_two_deep_hundred_wide','FC_three_deep_hundred_wide']   
reg_vec=np.logspace(-8,2,10)
lr_vec=np.logspace(-6,0,10)
n_val=4
n_cv=4

model_labels=['Linear','NonLin-1','NonLin-2','NonLin-3']
alpha_vec=[0.25,0.5,0.75,1.0]
width=0.2

for i in range(len(task_vec)):
    neu=0
    perf_all=nan*np.zeros((len(model_vec),len(files_vec)*n_hidden))
    for hh in files_vec:
        print (hh)
        for ii in range(len(model_vec)):
            perf_reg=nan*np.zeros((3,len(reg_vec),len(lr_vec)))  
            for iii in range(len(reg_vec)):
                apn=open(path_load+'performance_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f_n_test_%i_file_%i.pkl'%(loss_type,task_vec[i],model_vec[ii],iii,size_kernel,n_hidden,t_rnd_delay,sigma_noise,n_test,hh),'rb')
                perf_init=pkl.load(apn)['performance']
                apn.close()
                perf_init_pre=(1-perf_init[:,0]/perf_init[:,1])
                perf_init_pre[perf_init_pre==pinf]=nan
                perf_init_pre[perf_init_pre==minf]=nan
                perf_pre=np.nanmean(perf_init_pre,axis=(1,2))
                perf_reg[:,iii]=perf_pre
        
            index_max=np.where(perf_reg[1]==np.nanmax(perf_reg[1]))
            index_max_reg=index_max[0][0]
            index_max_lr=index_max[1][0]
            print (index_max)
        
            apn=open(path_load+'performance_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f_n_test_%i_file_%i.pkl'%(loss_type,task_vec[i],model_vec[ii],index_max_reg,size_kernel,n_hidden,t_rnd_delay,sigma_noise,n_test,hh),'rb')
            perf_init=pkl.load(apn)['performance']
            apn.close()
            perf_init_pre=(1-perf_init[:,0]/perf_init[:,1])
            perf_init_pre[perf_init_pre==pinf]=nan
            perf_init_pre[perf_init_pre==minf]=nan
            perf_pre=np.nanmean(perf_init_pre,axis=(2))
            perf_all[ii,neu:(neu+n_hidden)]=perf_pre[2,:,index_max_lr]
        neu=(neu+n_hidden)
        
    perf_all_m=np.nanmean(perf_all,axis=1)
    perf_all_sem=sem(perf_all,axis=1,nan_policy='omit')
    print (perf_all_m)
    print (perf_all_sem)

    fig=plt.figure(figsize=(2,2))
    ax=fig.add_subplot(111)
    functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.set_ylim([0.07,0.20])
    #ax.set_yticks([0.1,0.2,0.3])
    ax.set_xlim([-0.7,0.7])
    for jj in range(len(model_vec)):
       ax.bar(jj*width-1.5*width,perf_all_m[jj],yerr=perf_all_sem[jj],color='purple',width=width,alpha=alpha_vec[jj])
    ax.set_ylabel('$R^{2}$')
    fig.savefig('/home/ramon/Dropbox/chris_randy/plots/encodig_models_ANN_%s.pdf'%(task_vec[i]),dpi=500,bbox_inches='tight')        
    
# #####################################################################

# # Linear


# # XOR                                                                                                                                                                                                    
# fig=plt.figure(figsize=(2,2))
# ax=fig.add_subplot(111)
# functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.set_ylim([0.05,0.20])
# #ax.set_yticks([0.1,0.2,0.3])
# ax.set_xlim([-3.5*width,3.5*width])
# for i in range(len(model_vec)):
#    ax.bar(i*width-1.5*width,perf_all_m[1,i],yerr=perf_all_sem[1,i],color='purple',width=width,alpha=alpha_vec[i])
# ax.set_ylabel('$R^{2}$')
# fig.savefig('/home/ramon/Dropbox/chris_randy/plots/encoding_models_ANN_2_new2.pdf',dpi=500,bbox_inches='tight')
# 
######################################################################################       
#model_labels=['Linear','NonLin-1','NonLin-2','NonLin-3']
#alpha_vec=[0.4,0.6,0.8,1.0]
#width=0.30
#        
## Linear                                                                                                                                                                                                    
#fig=plt.figure(figsize=(2,2))
#ax=fig.add_subplot(111)
#functions_miscellaneous.adjust_spines(ax,['left','bottom'])
#ax.plot([-3.5*width,3.5*width],0.5*np.ones(2),color='black',linestyle='--')
#ax.set_ylim([0.4,1.0])
#ax.set_xlim([-3.5*width,3.5*width])
#plt.xticks(width*np.arange(len(model_vec))-1.5*width,model_labels,rotation='vertical')
#for i in range(len(model_vec)):
#   ax.bar(i*width-1.5*width,perf_all[0,i],color='green',width=width,alpha=alpha_vec[i])
#ax.set_ylabel('Decoding Performance')
#fig.savefig('/home/ramon/Dropbox/chris_randy/plots/behavior_tasks_ANN_1.pdf',dpi=500,bbox_inches='tight')
#
## XOR                                                                                                                                                                                                       
#fig=plt.figure(figsize=(2,2))
#ax=fig.add_subplot(111)
#functions_miscellaneous.adjust_spines(ax,['left','bottom'])
#ax.plot([-3.5*width,3.5*width],0.5*np.ones(2),color='black',linestyle='--')
#ax.set_ylim([0.4,1.0])
#ax.set_xlim([-3.5*width,3.5*width])
#plt.xticks(width*np.arange(len(model_vec))-1.5*width,model_labels,rotation='vertical')
#for i in range(len(model_vec)):
#   ax.bar(i*width-1.5*width,perf_all[1,i],color='green',width=width,alpha=alpha_vec[i])
#ax.set_ylabel('Decoding Performance')
#fig.savefig('/home/ramon/Dropbox/chris_randy/plots/behavior_tasks_ANN_2.pdf',dpi=500,bbox_inches='tight')
#
#        
#        
