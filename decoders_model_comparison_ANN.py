import numpy as np
import pickle as pkl
import nn_pytorch_encoders
import torch
import os
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
import functions_miscellaneous
from scipy.stats import sem
import scipy
import pandas
nan=float('nan')
minf=float('-inf')
pinf=float('inf')

###################################################################################################
# Parameters
t_steps=30
t_dec=20
stim_nat='Poisson'
dim_in=3
n_hidden=60
t_rnd_delay=10
sigma_test=1
n_test=40
files_vec=[0,2,3,4]

abs_path='/home/ramon/Dropbox/chris_randy/data_behavior/data_ANN/'

task_vec=['linear','2d_xor','xor']
model_vec=['FC_trivial','FC_one_deep_hundred_wide','FC_two_deep_hundred_wide','FC_three_deep_hundred_wide']
reg_vec=np.logspace(-8,2,10)                                                                                                                                                               
lr_vec=np.logspace(-6,0,10)
n_val=4
n_cv=4

alph=[0.25,0.5,0.75,1]
lab=['Linear','NonLin1','NonLin2','NonLin3']
width=0.2

for i in range(len(task_vec)):
   print (task_vec[i])
   perf_all=nan*np.zeros((len(files_vec),len(model_vec)))
   for hh in range(len(files_vec)):
      print ('  ',hh)
      for ii in range(len(model_vec)):
         print ('    ',model_vec[ii])
         perf_reg=nan*np.zeros((3,len(reg_vec),len(lr_vec)))
         for iii in range(len(reg_vec)):
            apn=open(abs_path+'performance_decoders_%s_%s_%i_hidden_%i_delay_%i_noise_%.1f_file_%i.pkl'%(task_vec[i],model_vec[ii],iii,n_hidden,t_rnd_delay,sigma_test,files_vec[hh]),'rb')
            perf_init=pkl.load(apn)['performance']
            apn.close()
            perf_pre=np.nanmean(perf_init,axis=1)
            perf_reg[:,iii]=perf_pre

         index_max=np.where(perf_reg[1]==np.nanmax(perf_reg[1]))
         print ('      ',index_max)
         index_max_reg=index_max[0][0]
         index_max_lr=index_max[1][0]
         
         apn=open(abs_path+'performance_decoders_%s_%s_%i_hidden_%i_delay_%i_noise_%.1f_file_%i.pkl'%(task_vec[i],model_vec[ii],index_max_reg,n_hidden,t_rnd_delay,sigma_test,files_vec[hh]),'rb')
         perf_init=np.mean(pkl.load(apn)['performance'],axis=1)
         apn.close()
         perf_all[hh,ii]=(perf_init[2,index_max_lr])

   perf_all_m=np.nanmean(perf_all,axis=0)
   perf_all_sem=sem(perf_all,axis=0,nan_policy='omit')
   print (perf_all_m)
   print (perf_all_sem)

   # Plots
   fig=plt.figure(figsize=(2,2))
   ax=fig.add_subplot(1,1,1)
   functions_miscellaneous.adjust_spines(ax,['left','bottom'])
   ax.set_ylabel('Decoding Performance')
   ax.set_ylim([0.45,0.85])
   ax.set_yticks([0.5,0.6,0.7,0.8])
   ax.set_xlim([-0.7,0.7])
   ax.plot(np.arange(3)-1,0.5*np.ones(3),color='black',linestyle='--')
   #ax.set_title('%s'%task_vec[i])
   for jj in range(len(model_vec)):
      ax.bar(jj*width-1.5*width,perf_all_m[jj],yerr=perf_all_sem[jj],width=width,color='green',label=lab[jj],alpha=alph[jj])   
   fig.savefig('/home/ramon/Dropbox/chris_randy/plots/decoders_models_ANN_%s_new.pdf'%(task_vec[i]),bbox_inches='tight',dpi=500) 

