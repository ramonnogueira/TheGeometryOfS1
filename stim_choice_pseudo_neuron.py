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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Normalize firing rate for eac time step
def normalize_fr(fr):
   fr_norm=np.zeros(np.shape(fr))
   for i in range(len(fr[0])):
      for ii in range(len(fr[0,0])):
         mean=np.mean(fr[:,i,ii])
         std=np.std(fr[:,i,ii])
         fr_norm[:,i,ii]=(fr[:,i,ii]-mean)/std
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
num_steps=int(np.round((dic_time['end_window']-dic_time['start_window'])/dic_time['resol']))
xx=np.arange(num_steps)*dic_time['resol']+dic_time['start_window']+dic_time['resol']/2.0

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
perc_tr=0.8
n_tr=int(perc_tr*nt_used)
n_te=int(nt_used-n_tr)
n_cv=100

for i in range(len(files_vec)):
   print (files_vec[i])
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
   firing_rate=functions_miscellaneous.create_firing_rate(rwin_time=neural_timings['rwin_nbase'],spikes_raw=spikes['time_stamps'],neu_ident=spikes['cluster'],dic_time=dic_time)[index_lick]
   firing_norm=normalize_fr(fr=firing_rate)
   neu_f=len(firing_norm[0])

   pop_train=nan*np.zeros((n_cv,4*n_tr,num_steps*neu_f))
   pop_test=nan*np.zeros((n_cv,4*n_te,num_steps*neu_f))

   stimulus=behavior['stimulus']
   choice=behavior['choice']
   reward=behavior['reward']
   index11=np.where((stimulus==1)&(choice==1))[0]
   index10=np.where((stimulus==1)&(choice==-1))[0]
   index01=np.where((stimulus==-1)&(choice==1))[0]
   index00=np.where((stimulus==-1)&(choice==-1))[0]
   index_all=[index11,index10,index01,index00]
   
   for j in range(len(index_all)):
      gg=-1
      skf=ShuffleSplit(n_splits=n_cv,train_size=perc_tr)
      for train, test in skf.split(index_all[j]):
         gg=(gg+1)
         ind_tr=np.random.choice(index_all[j][train],n_tr,replace=True)
         ind_te=np.random.choice(index_all[j][test],n_te,replace=True)
         pop_train[gg,j*n_tr:(j+1)*n_tr]=np.reshape(firing_norm[ind_tr],(n_tr,-1))
         pop_test[gg,j*n_te:(j+1)*n_te]=np.reshape(firing_norm[ind_te],(n_te,-1))

   if (i==0):
      pseudo_pop_train=pop_train.copy()
      pseudo_pop_test=pop_test.copy()
   else:
      pseudo_pop_train=np.concatenate((pseudo_pop_train,pop_train),axis=2)
      pseudo_pop_test=np.concatenate((pseudo_pop_test,pop_test),axis=2)

stimulus_train=nan*np.zeros((n_cv,4*n_tr))
stimulus_test=nan*np.zeros((n_cv,4*n_te))
choice_train=nan*np.zeros((n_cv,4*n_tr))
choice_test=nan*np.zeros((n_cv,4*n_te))
stim_vec=np.array([1,1,0,0])
choi_vec=np.array([1,0,1,0])
for j in range(4):
   for jj in range(n_cv):
      stimulus_train[jj,j*n_tr:(j+1)*n_tr]=stim_vec[j]*np.ones(n_tr)
      stimulus_test[jj,j*n_te:(j+1)*n_te]=stim_vec[j]*np.ones(n_te)
      choice_train[jj,j*n_tr:(j+1)*n_tr]=choi_vec[j]*np.ones(n_tr)
      choice_test[jj,j*n_te:(j+1)*n_te]=choi_vec[j]*np.ones(n_te)
   
      
perf_stim=nan*np.zeros((n_cv,2))
perf_choi=nan*np.zeros((n_cv,2))
for k in range(n_cv):
   #supp_st=LinearSVC(C=1,class_weight='balanced')
   supp_st=LogisticRegression(C=1e-2,class_weight='balanced')
   mod_st=supp_st.fit(pseudo_pop_train[k],stimulus_train[k])
   perf_stim[k,0]=supp_st.score(pseudo_pop_train[k],stimulus_train[k])
   perf_stim[k,1]=supp_st.score(pseudo_pop_test[k],stimulus_test[k])
   #
   #supp_ch=LinearSVC(C=1,class_weight='balanced')
   supp_ch=LogisticRegression(C=1e-2,class_weight='balanced')
   mod_ch=supp_ch.fit(pseudo_pop_train[k],choice_train[k])
   perf_choi[k,0]=supp_ch.score(pseudo_pop_train[k],choice_train[k])
   perf_choi[k,1]=supp_ch.score(pseudo_pop_test[k],choice_test[k])

print (np.mean(perf_stim,axis=0))
print (np.mean(perf_choi,axis=0))
