import os
import matplotlib.pylab as plt
import numpy as np
import scipy
import math
import sys
import tables
import pandas
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import licking_preprocessing
import functions_miscellaneous
from sklearn.svm import LinearSVC
from scipy.stats import ortho_group 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def create_xor_median(feat,v_used,m_orto):
    feat_sum=np.sum(feat[:,v_used],axis=2)
    feat_noise=(feat_sum+np.random.normal(0,1e-3,(np.shape(feat_sum))))
    n_feat=len(v_used)

    #m_orto=ortho_group.rvs(dim=n_feat)
    feat_trans=np.dot(feat_noise,m_orto)

    feat_binary=nan*np.zeros(np.shape(feat_trans))
    feat_binary_orig=nan*np.zeros(np.shape(feat_trans))
    for j in range(n_feat):
        # Rotated Space
        median=np.median(feat_trans[:,j])
        feat_binary[:,j][feat_trans[:,j]>median]=1
        feat_binary[:,j][feat_trans[:,j]<=median]=0
        # Original Space
        median_orig=np.median(feat_noise[:,j])
        feat_binary_orig[:,j][feat_noise[:,j]>median_orig]=1
        feat_binary_orig[:,j][feat_noise[:,j]<=median_orig]=0
    xor=np.sum(feat_binary,axis=1)%2
    lin=feat_binary[:,0]

    dic={}
    dic['feat_binary']=feat_binary
    dic['feat_binary_orig']=feat_binary_orig
    dic['xor']=xor
    dic['lin']=lin
    return dic

# Normalize for the whole trial
def normalize_fr(fr):
   fr_norm=np.zeros(np.shape(fr))
   for i in range(len(fr[0])):
       mean=np.mean(fr[:,i])
       std=np.std(fr[:,i])
       fr_norm[:,i]=(fr[:,i]-mean)/std
   fr_norm[np.isnan(fr_norm)]=0.0
   return fr_norm
##################################################

task_type='discrimination'
# Discrimination: 10 mice
# Detection: 4 mice

# Path Loads
abs_path_pre='/home/ramon/Dropbox/chris_randy/data/'
abs_path='/home/ramon/Dropbox/chris_randy/data/data_%s/sessions/'%task_type

################
# Extract Files
files_all=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_include_session_df')).loc[task_type]
files_neuro=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_big_waveform_info_df')) 
files_good=files_all[files_all['bad_perf']==False]
files_vec_pre=files_good[files_good['neural']==True].index.values
files_vec=[]
for i in files_vec_pre:
   try:
      files_neuro.loc[i[1]]
      files_vec.append(i[1])
   except:
      None
files_vec=np.array(files_vec)
#################

dic_time={}
dic_time['time_lock']='response' # response or stop_move
dic_time['start_window']=-2.0
dic_time['end_window']=0.0
dic_time['resol']=0.2
num_steps_pre=int(np.round((dic_time['end_window']-dic_time['start_window'])/dic_time['resol']))

# Lick temporal resolution. No tocar!!
dt_lick={}
dt_lick['time_lock']='response' # response or stop_move
dt_lick['start_window']=0.0
dt_lick['end_window']=0.5
dt_lick['resol']=0.5

quantities={}
quant_vec_ct=np.array(['contacts'])#'contacts'#,'tip_x','tip_y','fol_x','fol_y','angle'])
quant_vec_an=np.array([])#'angle','tip_x','tip_y','fol_x','fol_y'])
quant_vec_all=np.concatenate((quant_vec_ct,quant_vec_an))
quantities['contacts']=quant_vec_ct
quantities['analog']=quant_vec_an

features_slow=np.array([])
opto_dic=functions_miscellaneous.extract_opto(abs_path_pre)
###################################################

nt_used=500
v_used=np.array([[1,2],[1,3],[2,3]])
n_rand=20
t_ini=0
reg=1
n_null=1000
num_steps=(num_steps_pre-t_ini)
pertur_vec=np.logspace(-2,0.6,10)
#print (pertur_vec)

task_uq=np.array([[0,0],[0,1],[1,0],[1,1]])
task_uq_other=np.array([[0,0],[0,1],[1,1],[1,0]])
task=nan*np.zeros((4*nt_used,2))
task_lin=nan*np.zeros((4*nt_used))
task_xor=nan*np.zeros((4*nt_used))
index_perm_all=nan*np.zeros((n_null,4,5840))
for jj in range(4): # Loop on index 0 and 1 for a given task
    task[jj*nt_used:(jj+1)*nt_used]=np.outer(np.ones(nt_used),task_uq[jj])
    task_lin[jj*nt_used:(jj+1)*nt_used]=task_uq_other[jj,0]*np.ones(nt_used)
    task_xor[jj*nt_used:(jj+1)*nt_used]=task_uq_other[jj,1]*np.ones(nt_used)
    for jjj in range(n_null):
        index_perm_all[jjj,jj]=np.random.permutation(np.arange(5840))

perf_abs=nan*np.zeros((n_rand,len(v_used),2,2))
perf_abs_pert=nan*np.zeros((n_rand,len(v_used),len(pertur_vec),2,2))
perf_abs_null=nan*np.zeros((n_rand,len(v_used),n_null,2,2))
perf_lin_null=nan*np.zeros((n_rand,len(v_used),n_null,4))
perf_xor_null=nan*np.zeros((n_rand,len(v_used),n_null,4))
for r in range(n_rand):
    print (r)
    for i in range(len(files_vec)):
        #print (files_vec[i])
        analog=functions_miscellaneous.extract_analog(abs_path+files_vec[i])
        contact=functions_miscellaneous.extract_contacts(abs_path+files_vec[i])
        index_use=functions_miscellaneous.extract_index_use(opto_dic[files_vec[i]],abs_path+files_vec[i])
        timings=functions_miscellaneous.extract_timings(index_use,abs_path+files_vec[i])
        licks=functions_miscellaneous.extract_licks(abs_path+files_vec[i])
        neural_timings=functions_miscellaneous.extract_neural_timings(index_use,abs_path+files_vec[i])
        spikes=functions_miscellaneous.extract_spikes(abs_path+files_vec[i])
   
        # Esto lick rate es para sacar index_lick
        lick_rate_prepre=functions_miscellaneous.create_licks(reference_time=timings['rwin_sec'],licking_time=licks['licking_time'],lick_side=licks['lick_side'],dic_time=dt_lick)
        index_lick=(np.sum(abs(lick_rate_prepre),axis=1)!=0)
   
        behavior=functions_miscellaneous.extract_behavior(index_use,index_lick,abs_path+files_vec[i])
        feat_pre=functions_miscellaneous.create_feat(quantities=quantities,timings=timings,contact=contact,analog=analog,dic_time=dic_time)[index_lick][:,0]
        firing_pre=functions_miscellaneous.create_firing_rate(rwin_time=neural_timings['rwin_nbase'],spikes_raw=spikes['time_stamps'],neu_ident=spikes['cluster'],dic_time=dic_time)[index_lick]
        firing_rate=normalize_fr(fr=firing_pre[:,:,t_ini:]) #Cuidado!
        neu_f=len(firing_rate[0])
        pop=nan*np.zeros((len(v_used),4*nt_used,num_steps*neu_f))
        
        for h in range(len(v_used)):
            feat_sum=np.sum(feat_pre[:,v_used[h]],axis=(1,2)) # Cuidado!!
            index_ct=(feat_sum!=0)
            feat_used=feat_pre[index_ct]
            firing_used=firing_rate[index_ct]

            create_xor=create_xor_median(feat=feat_used,v_used=v_used[h],m_orto=np.array([[1,0],[0,1]]))
            feat_binary=create_xor['feat_binary_orig']
            feat_uq=np.unique(feat_binary,axis=0)
               
            for jj in range(len(feat_uq)):
                index_jj=np.where((feat_binary[:,0]==feat_uq[jj,0])&(feat_binary[:,1]==feat_uq[jj,1]))[0]
                ind_samp=np.random.choice(index_jj,nt_used,replace=True)
                pop[h,jj*nt_used:(jj+1)*nt_used]=np.reshape(firing_used[ind_samp],(nt_used,-1))
                
        if (i==0):
            pseudo=pop.copy()
        else:
            pseudo=np.concatenate((pseudo,pop),axis=2)

    # Decoding    
    for h in range(len(v_used)):
        print ('  ',h)
        perf_abs[r,h]=functions_miscellaneous.abstraction_2D(feat_decod=pseudo[h],feat_binary=task,reg=reg)[:,:,1]
        for tt in range(n_null):
            pseudo_null=nan*np.zeros(np.shape(pseudo[h]))
            for hhh in range(len(task_uq)):
                index_hhh=np.where((task[:,0]==task_uq[hhh,0])&(task[:,1]==task_uq[hhh,1]))[0]
                index_perm=np.array(index_perm_all[tt,hhh],dtype=np.int16)
                pseudo_null[index_hhh]=pseudo[h,index_hhh][:,index_perm]
            perf_abs_null[r,h,tt]=functions_miscellaneous.abstraction_2D(feat_decod=pseudo_null,feat_binary=task,reg=reg)[:,:,1]

        # for hh in range(len(pertur_vec)):
        #     pseudo_pert=nan*np.zeros(np.shape(pseudo[h]))
        #     for hhh in range(len(task_uq)):
        #         index_hhh=np.where((task[:,0]==task_uq[hhh,0])&(task[:,1]==task_uq[hhh,1]))[0]
        #         pert=np.random.normal(0,pertur_vec[hh],(len(pseudo[0,0])))
        #         pseudo_pert[index_hhh]=(pseudo[h,index_hhh]+pert)
        #     perf_abs_pert[r,h,hh]=functions_miscellaneous.abstraction_2D(feat_decod=pseudo_pert,feat_binary=task,reg=reg)[:,:,1]

perf_abs_pre=np.nanmean(perf_abs,axis=(1,2,3))
perf_abs_m=np.nanmean(perf_abs_pre,axis=(0))
# perf_abs_err=np.nanstd(perf_abs_pre,axis=(0))
# perf_abs_ind=np.nanmean(perf_abs,axis=3)
# perf_abs_c1=0.5*(perf_abs_ind[:,0,0]+perf_abs_ind[:,1,0])
# perf_abs_c2=0.5*(perf_abs_ind[:,0,1]+perf_abs_ind[:,2,0])
# perf_abs_c3=0.5*(perf_abs_ind[:,1,1]+perf_abs_ind[:,2,1])
# perf_c1_m=np.nanmean(perf_abs_c1)
# perf_c1_err=np.std(perf_abs_c1)
# perf_c2_m=np.nanmean(perf_abs_c2)
# perf_c2_err=np.std(perf_abs_c2)
# perf_c3_m=np.nanmean(perf_abs_c3)
# perf_c3_err=np.std(perf_abs_c3)

perf_abs_null_pre=np.nanmean(perf_abs_null,axis=(1,3,4))
perf_abs_null_m=np.nanmean(perf_abs_null_pre,axis=(0))

# perf_abs_pre_null=np.nanmean(perf_abs_null,axis=(1,3,4))
# perf_abs_null_m=np.nanmean(perf_abs_pre_null,axis=(0))
# perf_abs_null_err=np.nanstd(perf_abs_pre_null,axis=(0))
# perf_abs_ind_null=np.nanmean(perf_abs_null,axis=4)
# perf_abs_c1_null=0.5*(perf_abs_ind_null[:,0,:,0]+perf_abs_ind_null[:,1,:,0])
# perf_abs_c2_null=0.5*(perf_abs_ind_null[:,0,:,1]+perf_abs_ind_null[:,2,:,0])
# perf_abs_c3_null=0.5*(perf_abs_ind_null[:,1,:,1]+perf_abs_ind_null[:,2,:,1])
# perf_c1_null_m=np.nanmean(perf_abs_c1_null,axis=0)
# perf_c1_null_err=np.std(perf_abs_c1_null,axis=0)
# perf_c2_null_m=np.nanmean(perf_abs_c2_null,axis=0)
# perf_c2_null_err=np.std(perf_abs_c2_null,axis=0)
# perf_c3_null_m=np.nanmean(perf_abs_c3_null,axis=0)
# perf_c3_null_err=np.std(perf_abs_c3_null,axis=0)

#####################################
# Plots
# width=0.2

# fig=plt.figure(figsize=(3,2))
# ax=fig.add_subplot(1,1,1)
# functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.set_ylim([0.4,1.0])
# ax.set_ylabel('Decoding Performance')
# ax.bar(0,perf_abs_m,yerr=perf_abs_err,width=width,color='green',alpha=0.7)
# ax.scatter(0,perf_c1_m,s=10,color='blue',edgecolor='black',linestyle='None')
# ax.scatter(0,perf_c2_m,s=10,color='green',edgecolor='black',linestyle='None')
# ax.scatter(0,perf_c3_m,s=10,color='red',edgecolor='black',linestyle='None')
# ax.plot(np.linspace(-0.5,0.5,10),0.5*np.ones(10),color='black',linestyle='--')
#fig.savefig('/home/ramon/Dropbox/chris_randy/plots/lin_xor_abs_recordings.pdf',dpi=500,bbox_inches='tight')

# Plot as a function of Perturbation Strength
# fig=plt.figure(figsize=(3,3))
# ax=fig.add_subplot(1,1,1)
# functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.set_ylim([0.4,1.0])
# ax.set_ylabel('Decoding Performance')
# ax.set_xlabel('Perturbation Strength ($\sigma$)')
# ax.plot(pertur_vec,perf_abs_null_m,color='green')
# ax.fill_between(pertur_vec,perf_abs_null_m-perf_abs_null_err,perf_abs_null_m+perf_abs_null_err,color='green',alpha=0.5)
# ax.plot(pertur_vec,0.5*np.ones(len(pertur_vec)),color='black',linestyle='--')
# ax.set_xscale('log')
# #ax.set_xticks([0.01,0.1,1])
# fig.savefig('/home/ramon/Dropbox/chris_randy/plots/abs_perturbation.pdf',dpi=500,bbox_inches='tight')
