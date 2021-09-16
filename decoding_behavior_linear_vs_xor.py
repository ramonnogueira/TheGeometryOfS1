import os
import numpy as np
import matplotlib.pyplot as plt
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
from scipy.stats import ortho_group

import nn_pytorch_encoders
import features_whiskers
import spikes_processing
import licking_preprocessing
import decoders_classes
import functions_miscellaneous

def norm_sum(feat):
   feat_norm=nan*np.zeros(np.shape(feat))
   for i in range(len(feat[0])):
      fm=np.nanmean(feat[:,i])
      fstd=np.nanstd(feat[:,i])
      feat_norm[:,i]=(feat[:,i]-fm)/fstd
   return feat_norm
      
#############################

nan=float('nan')

task_type='discrimination'
model_vec=['FC_trivial','FC_one_deep_hundred_wide','FC_two_deep_hundred_wide','FC_three_deep_hundred_wide']
n_val=4
n_cv=4
lr_vec_nn=np.logspace(-4,1,10)
reg_vec_nn=np.logspace(-3,3,10)
index_xor=[1,2,3]
std_xor=0.02
n_orto=500
reg=1

print ('std ',std_xor)
print ('n orto ',n_orto)

# Path Loads
abs_path_pre='/home/ramon/Dropbox/chris_randy/data/'
abs_path='/home/ramon/Dropbox/chris_randy/data/data_%s/sessions/'%task_type

# Path save
abs_path_save='/home/ramon/Dropbox/chris_randy/data_behavior/xor'

################
# Extract Files
files_all=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_include_session_df')).loc[task_type]
files_good=files_all[files_all['bad_perf']==False]
files_vec=files_good.index.values

mice_pre=[]
for i in files_vec:
   mice_pre.append(i[0])
mice_vec=np.unique(mice_pre)

print (mice_vec)
#################

dic_time={}
dic_time['time_lock']='response' # response or stop_move
dic_time['start_window']=-2.0
dic_time['end_window']=0.0
dic_time['resol']=0.1
num_steps=int((dic_time['end_window']-dic_time['start_window'])/dic_time['resol'])

print ('resol ',dic_time['resol'])

# Lick temporal resolution. No tocar!!
dt_lick={}
dt_lick['time_lock']='response' # response or stop_move
dt_lick['start_window']=0.0
dt_lick['end_window']=0.5
dt_lick['resol']=0.05

quantities={}
quant_vec_ct=np.array(['contacts'])#,'tip_x','tip_y','fol_x','fol_y','angle'])
quant_vec_an=np.array([])#'angle','tip_x','tip_y','fol_x','fol_y'])
quant_vec_all=np.concatenate((quant_vec_ct,quant_vec_an))
quantities['contacts']=quant_vec_ct
quantities['analog']=quant_vec_an

features_slow=np.array([])

###########################################################################################

opto_dic=functions_miscellaneous.extract_opto(abs_path_pre)

perf_lin=nan*np.zeros((len(mice_vec),len(model_vec)))
perf_xor=nan*np.zeros((len(mice_vec),len(model_vec)))
for k in range(len(mice_vec)):
   print (mice_vec[k])
   files_mouse=files_good.loc[mice_vec[k]].index.values
   for i in range(len(files_mouse)):
      print ('  ',files_mouse[i])
      analog=functions_miscellaneous.extract_analog(abs_path+files_mouse[i])
      contact=functions_miscellaneous.extract_contacts(abs_path+files_mouse[i])
      index_use=functions_miscellaneous.extract_index_use(opto_dic[files_mouse[i]],abs_path+files_mouse[i])
      timings=functions_miscellaneous.extract_timings(index_use,abs_path+files_mouse[i])
      licks=functions_miscellaneous.extract_licks(abs_path+files_mouse[i])

      # Esto lick rate es para sacar index_lick
      lick_rate_prepre=functions_miscellaneous.create_licks(reference_time=timings['rwin_sec'],licking_time=licks['licking_time'],lick_side=licks['lick_side'],dic_time=dt_lick)
      index_lick=(np.sum(abs(lick_rate_prepre),axis=1)!=0)
      # Este es el actual lick rate
      lick_rate_pre=functions_miscellaneous.create_licks(reference_time=timings['rwin_sec'],licking_time=licks['licking_time'],lick_side=licks['lick_side'],dic_time=dic_time)
      lick_rate=lick_rate_pre[index_lick]

      behavior=functions_miscellaneous.extract_behavior(index_use,index_lick,abs_path+files_mouse[i])
      feat_pre=functions_miscellaneous.create_feat(quantities=quantities,timings=timings,contact=contact,analog=analog,dic_time=dic_time)[index_lick]

      if i==0:
         all_super={}
         all_super['stimulus']=behavior['stimulus']
         all_super['choice']=behavior['choice']
         all_super['reward']=behavior['reward']
         all_super['position']=behavior['position']
         all_super['difficulty']=behavior['difficulty']
         all_super['feat']=feat_pre
      else:
         all_super=functions_miscellaneous.create_super_mouse(all_super=all_super,behavior=behavior,feat=feat_pre)

   feat=(all_super['feat'][:,0,index_xor]*dic_time['resol'])
   print (np.shape(feat))
   feat_noise=(feat+np.random.normal(0,std_xor,np.shape(feat)))
   feat_sum=np.sum(feat,axis=2)
   feat_sum_noise=np.sum(feat_noise,axis=2)
   m_orto=functions_miscellaneous.best_orto_rotation(feat=feat_sum_noise,n=n_orto,reg=reg,reg_vec=reg_vec_nn,n_cv=n_cv,n_val=n_val,validation=False)
   feat_sum_noise_rot=np.dot(feat_sum_noise,m_orto)
   feat_binary=nan*np.zeros(np.shape(feat_sum_noise))
   for p in range(len(index_xor)):
      median_ct=np.median(feat_sum_noise_rot[:,p])
      feat_binary[:,p]=(feat_sum_noise_rot[:,p]>=median_ct)
   lin=feat_binary[:,0]
   xor=np.sum(feat_binary,axis=1)%2    
   feat_noise_flat=np.reshape(feat_noise,(len(feat_noise),len(feat_noise[0])*len(feat_noise[0,0])))

   #################################
   # Scatter Plots
   c1_ct=feat_sum_noise[:,0]
   c1_med=np.median(c1_ct)
   c3_ct=feat_sum_noise[:,2]
   c3_med=np.median(c3_ct)
   #
   fig=plt.figure(figsize=(1.5,1.5))
   ax=fig.add_subplot(1,1,1)
   functions_miscellaneous.adjust_spines(ax,['left','bottom'])
   ax.set_xlabel('C1 Contacts')
   ax.set_ylabel('C3 Contacts')
   ax.set_xticks([0,10])
   ax.set_yticks([0,10])
   ax.scatter(c1_ct[c3_ct>c3_med],c3_ct[c3_ct>c3_med],color='orange',s=0.25)
   ax.scatter(c1_ct[c3_ct<c3_med],c3_ct[c3_ct<c3_med],color='green',s=0.25)
   fig.savefig('/home/ramon/Dropbox/chris_randy/plots/scatter_contacts_linear_xor_%s.pdf'%mice_vec[k],dpi=500,bbox_inches='tight')
   
   # for j in range(len(model_vec)):
#       print (model_vec[j])
      
#       # Task Linear
#       print ('Task Linear')
#       cl=functions_miscellaneous.neural_net(feat=feat_noise_flat,clase=lin,reg_vec=reg_vec_nn,lr_vec=lr_vec_nn,n_val=n_val,n_cv=n_cv)
#       nn=cl.nn(model=model_vec[j])
#       perf_test=nn['performance'][1]
#       perf_val=nn['performance'][2]
#       index_max=np.where(perf_test==np.nanmax(perf_test))
#       perf_lin[k,j]=perf_val[index_max][0]
      
#       #Task XOR
#       print ('Task Xor')
#       cl=functions_miscellaneous.neural_net(feat=feat_noise_flat,clase=xor,reg_vec=reg_vec_nn,lr_vec=lr_vec_nn,n_val=n_val,n_cv=n_cv)
#       nn=cl.nn(model=model_vec[j])
#       perf_test=nn['performance'][1]
#       perf_val=nn['performance'][2]
#       index_max=np.where(perf_test==np.nanmax(perf_test))
#       perf_xor[k,j]=perf_val[index_max][0]

#       print (np.nanmean(perf_lin,axis=0))
#       print (np.nanmean(perf_xor,axis=0))

#    svm_cl=functions_miscellaneous.svm_sum_val(feat=feat_noise_flat,clase=lin,reg_vec=reg_vec_nn,n_val=n_val,n_cv=n_cv)
#    svm=svm_cl.svm(balanced=False)
#    perf_test=svm['performance'][1]
#    perf_val=svm['performance'][2]
#    index_max=np.where(perf_test==np.nanmax(perf_test))
#    print ('SKLEARN ',perf_val[index_max])

# perf_lin_m=np.nanmean(perf_lin,axis=0)
# perf_lin_sem=sem(perf_lin,axis=0,nan_policy='omit')
# perf_xor_m=np.nanmean(perf_xor,axis=0)
# perf_xor_sem=sem(perf_xor,axis=0,nan_policy='omit')

# ############################################################
# model_labels=['Linear','NonLin-1','NonLin-2','NonLin-3']
# alpha_vec=[0.4,0.6,0.8,1.0]
# width=0.30

# # Linear
# fig=plt.figure(figsize=(2,2))
# ax=fig.add_subplot(111)
# functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.plot([-3.5*width,3.5*width],0.5*np.ones(2),color='black',linestyle='--')
# ax.set_ylim([0.4,1.0])
# ax.set_xlim([-3.5*width,3.5*width])
# plt.xticks(width*np.arange(len(model_vec))-1.5*width,model_labels,rotation='vertical')
# for i in range(len(model_vec)):
#    ax.bar(i*width-1.5*width,perf_lin_m[i],yerr=perf_lin_sem[i],color='green',width=width,alpha=alpha_vec[i])
# ax.set_ylabel('Decoding Performance')
# fig.savefig('/home/ramon/Dropbox/chris_randy/figures/figure2/easy_vs_xor_1_2.pdf',dpi=500,bbox_inches='tight')

# # XOR
# fig=plt.figure(figsize=(2,2))
# ax=fig.add_subplot(111)
# functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.plot([-3.5*width,3.5*width],0.5*np.ones(2),color='black',linestyle='--')
# ax.set_ylim([0.4,1.0])
# ax.set_xlim([-3.5*width,3.5*width])
# plt.xticks(width*np.arange(len(model_vec))-1.5*width,model_labels,rotation='vertical')
# for i in range(len(model_vec)):
#    ax.bar(i*width-1.5*width,perf_xor_m[i],yerr=perf_xor_sem[i],color='green',width=width,alpha=alpha_vec[i])
# ax.set_ylabel('Decoding Performance')
# fig.savefig('/home/ramon/Dropbox/chris_randy/figures/figure2/easy_vs_xor_2_2.pdf',dpi=500,bbox_inches='tight')
   

# # ind1=np.where(lin==1)[0]
# # ind0=np.where(lin==0)[0]
# # fig=plt.figure(figsize=(1,1))
# # ax=fig.add_subplot(111)
# # functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# # ax.set_xlabel('C1 contacts')
# # ax.set_ylabel('C3 contacts')
# # ax.scatter(feat_sum_noise[ind1,0],feat_sum_noise[ind1,1],marker='.',linewidths=0,color=(0,0.5,0,1),s=5)
# # ax.scatter(feat_sum_noise[ind0,0],feat_sum_noise[ind0,1],marker='.',linewidths=0,color=(142/255,196/255,142/255),s=5)
# # fig.savefig('/home/ramon/Dropbox/chris_randy/figures/figure2/figure2e_graph_easy.pdf',dpi=500,bbox_inches='tight')
# # 
