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
# Dimensionality ANN
# Parameters
t_steps=20  
stim_nat='Poisson'
dim_in=3
n_hidden=40
t_rnd_delay=5
sigma_noise=0.5
size_kernel=10
loss_type='gaussian'
n_test=100

data_set='validation'
score_type='r2'

abs_path='/home/ramon/Documents/chris_randy/data_analysis/data_ANN/'

task_vec=['linear','xor']
model_vec=['FC_trivial','FC_one_deep_hundred_wide','FC_two_deep_hundred_wide','FC_three_deep_hundred_wide']
reg_vec=np.logspace(-5,0,5) 
lr_vec=np.logspace(-5,0,6) 
n_val=4
n_cv=4

perf_all=nan*np.zeros((len(task_vec),n_hidden,len(model_vec)))
for i in range(len(task_vec)):
   for ii in range(len(model_vec)):
      print (ii)
      perf_reg=nan*np.zeros((3,len(reg_vec),6))
      for iii in range(len(reg_vec)):
         #try:
         print ('  %i'%iii)
         #apn=open(abs_path+'performance_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f_n_test_%i.pkl'%(loss_type,task_vec[i],model_vec[ii],iii,size_kernel,n_hidden,t_rnd_delay,sigma_noise,n_test),'rb')
         apn=open(abs_path+'performance_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f.pkl'%(loss_type,task_vec[i],model_vec[ii],iii,size_kernel,n_hidden,t_rnd_delay,sigma_noise),'rb')
         perf_init=pkl.load(apn)['performance']
         apn.close()
         if score_type=='llh':
            perf_init_pre=(perf_init[:,0])
         if score_type=='r2':
            perf_init_pre=(1-perf_init[:,0]/perf_init[:,1])
         perf_pre=np.nanmean(perf_init_pre,axis=(1,2))
         perf_pre[perf_pre==minf]=nan
         perf_pre[perf_pre==pinf]=nan
         perf_reg[:,iii]=perf_pre
         
         #except:
         #   None

      if score_type=='llh':
         index_max=np.where(perf_reg[1]==np.nanmin(perf_reg[1]))
      if score_type=='r2':
         index_max=np.where(perf_reg[1]==np.nanmax(perf_reg[1]))     
      index_max_reg=index_max[0][0]
      index_max_lr=index_max[1][0]
         
      #apn=open(abs_path+'performance_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f_n_test_%i.pkl'%(loss_type,task_vec[i],model_vec[ii],index_max_reg,size_kernel,n_hidden,t_rnd_delay,sigma_noise,n_test),'rb')
      apn=open(abs_path+'performance_nonlinear_%s_%s_%s_%i_kernel_%i_hidden_%i_%i_noise_%.1f.pkl'%(loss_type,task_vec[i],model_vec[ii],index_max_reg,size_kernel,n_hidden,t_rnd_delay,sigma_noise),'rb')
      perf_init=pkl.load(apn)['performance']
      apn.close()
      if score_type=='llh':
         perf_init_pre=(perf_init[:,0])
      if score_type=='r2':
         perf_init_pre=(1-perf_init[:,0]/perf_init[:,1])   
      perf_pre=np.nanmean(perf_init_pre,axis=(2))
      perf_pre[perf_pre==minf]=nan
      perf_pre[perf_pre==pinf]=nan
      if data_set=='train':
         perf_all[i,:,ii]=(perf_pre[0,:,index_max_lr])
      if data_set=='test':
         perf_all[i,:,ii]=(perf_pre[1,:,index_max_lr])
      if data_set=='validation':
         perf_all[i,:,ii]=(perf_pre[2,:,index_max_lr])

perf_all_mean=np.nanmean(perf_all,axis=1)
perf_all_sem=sem(perf_all,axis=1,nan_policy='omit')

print (perf_all_mean)

# Plots
#col=['black','red','green','blue']
alph=[0.25,0.5,0.75,1]
lab=['Linear','NonLin1','NonLin2','NonLin3']
width=0.6

for i in range(len(task_vec)):
   fig=plt.figure(figsize=(3,2))
   ax=fig.add_subplot(1,1,1)
   functions_miscellaneous.adjust_spines(ax,['left','bottom'])
   if score_type=='llh':
      ax.set_ylabel('Loss (MSE)')
      #ax.set_ylim([0.65,2.5])
   if score_type=='r2':
      ax.set_ylabel('$R^{2}$')
      #ax.set_ylim([0.3,0.75])
      #ax.set_ylim([0.1,0.3])
   ax.set_xlim([-0.5,len(lab)-0.5])
   ax.set_ylim([0,0.7])
   #ax.plot(np.arange(7)-1,0.5*np.ones(7),color='black',linestyle='--')
   #ax.set_title('%s'%task_vec[i])
   plt.xticks(np.arange(len(lab)),lab)
   for ii in range(len(lab)):
      ax.bar(ii,perf_all_mean[i,ii],yerr=perf_all_sem[i,ii],width=width,color='black',alpha=alph[ii])

   print (scipy.stats.wilcoxon(perf_all[i,:,0],perf_all[i,:,1]))

   fig.savefig('/home/ramon/Documents/chris_randy/plots/encoders_models_ANN_%s_%s_%s_n_test_%i.pdf'%(task_vec[i],data_set,score_type,n_test),bbox_inches='tight',dpi=500)

#############################
# Fig 2

fig1=plt.figure(figsize=(5,3*3.5))
for i in range(len(task_vec)):
   ax1=fig1.add_subplot(3,1,i+1)
   ax1.scatter(perf_all[i,:,0],perf_all[i,:,1])
   ax1.plot([0,35],[0,35])

fig1.savefig('/home/ramon/Documents/chris_randy/plots/scatter_encoders_models_ANN_%s_%s_n_test_%i.png'%(data_set,score_type,n_test),bbox_inches='tight',dpi=500)
