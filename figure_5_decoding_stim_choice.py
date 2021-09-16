import os
import matplotlib.pylab as plt
import numpy as np
import scipy
import math
import sys
import pandas
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import licking_preprocessing
import functions_miscellaneous
from scipy.stats import ortho_group
from sklearn.linear_model import LogisticRegression
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
      
task_type='discrimination'
# Discrimination: 10 mice
# Detection: 4 mice

n_cv=4
n_val=4
reg_vec=np.logspace(-4,4,10)

# Path Loads
abs_path_pre='/home/ramon/Dropbox/chris_randy/data/'
abs_path='/home/ramon/Dropbox/chris_randy/data/data_%s/sessions/'%task_type

################
# Extract Files
files_all=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_include_session_df')).loc[task_type]
files_neuro=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_big_waveform_info_df')) # From Chris: use basically this file to choose neural sessions. Do not use L1 and L6b neurons
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
dic_time['end_window']=0.4
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
###############################################

n_decorr=10
perf_pre=nan*np.zeros((2,len(files_vec),num_steps))
print (files_vec)
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
    # Este es el actual lick rate                                                                                                                                                                       
    lick_rate_pre=functions_miscellaneous.create_licks(reference_time=timings['rwin_sec'],licking_time=licks['licking_time'],lick_side=licks['lick_side'],dic_time=dic_time)
    lick_rate=lick_rate_pre[index_lick]
    
    behavior=functions_miscellaneous.extract_behavior(index_use,index_lick,abs_path+files_vec[i])
    feat_pre=functions_miscellaneous.create_feat(quantities=quantities,timings=timings,contact=contact,analog=analog,dic_time=dic_time)[index_lick][:,0]
    firing_rate=functions_miscellaneous.create_firing_rate(rwin_time=neural_timings['rwin_nbase'],spikes_raw=spikes['time_stamps'],neu_ident=spikes['cluster'],dic_time=dic_time)[index_lick]
    neu_f=len(firing_rate[0])

    stimulus=behavior['stimulus']
    choice=behavior['choice']
    reward=behavior['reward']
    index_cor=np.where(reward==1)[0]
    index_inc=np.where(reward==-1)[0]

    for j in range(num_steps):
      print (j)  
      firing_resh=np.reshape(firing_rate[:,:,0:(j+1)],(len(firing_rate),-1))

      ####################################
      # Decode Stim
      perf_stim_reg_pre=nan*np.zeros((3,n_val,n_cv,n_decorr,len(reg_vec)))
      kf1=KFold(n_val)
      for g_val,index1 in enumerate(kf1.split(firing_resh)):
          train1=index1[0]
          val=index1[1]
          kf=KFold(n_cv)
          for g,index in enumerate(kf.split(firing_resh[train1])):
              train=index[0]
              test=index[1]
              values=nan*np.zeros(len(stimulus[train1][train]))
              values[(reward[train1][train]==-1)&(stimulus[train1][train]==-1)]=0
              values[(reward[train1][train]==-1)&(stimulus[train1][train]==1)]=1
              values[(reward[train1][train]==1)&(stimulus[train1][train]==-1)]=2
              values[(reward[train1][train]==1)&(stimulus[train1][train]==1)]=3
              values=np.array(values,dtype=np.int16)
              min_class=np.nanmin([len(np.where(values==0)[0]),len(np.where(values==1)[0]),len(np.where(values==2)[0]),len(np.where(values==3)[0])])
              for jj in range(n_decorr):
                  index_t=np.array([])
                  for tt in range(4):
                      index_t=np.concatenate((index_t,np.random.choice(np.where(values==tt)[0],size=min_class,replace=False)))
                  index_t=np.sort(index_t)
                  index_t=np.array(index_t,dtype=np.int16)
                  for jjj in range(len(reg_vec)):
                      mod=LogisticRegression(C=1/reg_vec[jjj])
                      mod.fit(firing_resh[train1][train][index_t],stimulus[train1][train][index_t])
                      perf_stim_reg_pre[0,g_val,g,jj,jjj]=mod.score(firing_resh[train1][train],stimulus[train1][train])
                      perf_stim_reg_pre[1,g_val,g,jj,jjj]=mod.score(firing_resh[train1][test],stimulus[train1][test])
                      perf_stim_reg_pre[2,g_val,g,jj,jjj]=mod.score(firing_resh[val],stimulus[val])
      perf_stim_reg=np.nanmean(perf_stim_reg_pre,axis=(1,2,3))
      index_max_stim=np.where(perf_stim_reg[1]==np.nanmax(perf_stim_reg[1]))[0][0]
      perf_pre[0,i,j]=perf_stim_reg[2,index_max_stim]

      ####################################
      # Decode Choice
      perf_choi_reg_pre=nan*np.zeros((3,n_val,n_cv,n_decorr,len(reg_vec)))
      kf1=KFold(n_val)
      for g_val,index1 in enumerate(kf1.split(firing_resh)):
          train1=index1[0]
          val=index1[1]
          kf=KFold(n_cv)
          for g,index in enumerate(kf.split(firing_resh[train1])):
              train=index[0]
              test=index[1]
              values=nan*np.zeros(len(choice[train1][train]))
              values[(reward[train1][train]==-1)&(choice[train1][train]==-1)]=0
              values[(reward[train1][train]==-1)&(choice[train1][train]==1)]=1
              values[(reward[train1][train]==1)&(choice[train1][train]==-1)]=2
              values[(reward[train1][train]==1)&(choice[train1][train]==1)]=3
              values=np.array(values,dtype=np.int16)
              min_class=np.nanmin([len(np.where(values==0)[0]),len(np.where(values==1)[0]),len(np.where(values==2)[0]),len(np.where(values==3)[0])])
              for jj in range(n_decorr):
                  index_t=np.array([])
                  for tt in range(4):
                      index_t=np.concatenate((index_t,np.random.choice(np.where(values==tt)[0],size=min_class,replace=False)))
                  index_t=np.sort(index_t)
                  index_t=np.array(index_t,dtype=np.int16)
                  for jjj in range(len(reg_vec)):
                      mod=LogisticRegression(C=1/reg_vec[jjj])
                      mod.fit(firing_resh[train1][train][index_t],choice[train1][train][index_t])
                      perf_choi_reg_pre[0,g_val,g,jj,jjj]=mod.score(firing_resh[train1][train],choice[train1][train])
                      perf_choi_reg_pre[1,g_val,g,jj,jjj]=mod.score(firing_resh[train1][test],choice[train1][test])
                      perf_choi_reg_pre[2,g_val,g,jj,jjj]=mod.score(firing_resh[val],choice[val])
      perf_choi_reg=np.nanmean(perf_choi_reg_pre,axis=(1,2,3))
      index_max_choi=np.where(perf_choi_reg[1]==np.nanmax(perf_choi_reg[1]))[0][0]
      perf_pre[1,i,j]=perf_choi_reg[2,index_max_choi]
      print (np.nanmean(perf_pre[1],axis=0))

      
####################################################
# Info stimulus and Choice
perf_sc_m=np.nanmean(perf_pre,axis=1)
perf_sc_sem=sem(perf_pre,axis=1)

labels=['Stimulus','Choice']
colors=['green','blue']
fig=plt.figure(figsize=(2,2))
ax=fig.add_subplot(111)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.axvline(0,color='black',linestyle='--')
ax.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
for i in range(2):
   ax.plot(xx,perf_sc_m[i],color=colors[i],label=labels[i])
   ax.fill_between(xx,perf_sc_m[i]-perf_sc_sem[i],perf_sc_m[i]+perf_sc_sem[i],color=colors[i],alpha=0.5)
plt.legend(loc='best')
ax.set_ylim([0.4,0.8])
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Decoding Performance')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/decoding_stim_choice_time_2.pdf',dpi=500,bbox_inches='tight')


