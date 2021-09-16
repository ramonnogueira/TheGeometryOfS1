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
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import licking_preprocessing
import functions_miscellaneous
nan=float('nan')

task_type='discrimination'
# Discrimination: 10 mice
# Detection: 4 mice

reg_vec=np.logspace(-7,3,20)
reg=1
n_val=4
n_cv=4
n_decorr=10

# Path Loads
abs_path_pre='/home/ramon/Dropbox/chris_randy/data/'
abs_path='/home/ramon/Dropbox/chris_randy/data/data_%s/sessions/'%task_type


################
# Extract Files
files_all=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_include_session_df')).loc[task_type]
files_good=files_all[(files_all['bad_perf']==False)]
files_vec_pre=files_good.index.values
files_vec=[]
for i in range(len(files_vec_pre)):
   if files_vec_pre[i][0]!='200CR':
      files_vec.append(files_vec_pre[i])
files_vec=np.array(files_vec)
      
mice_pre=[]
for i in files_vec:
   mice_pre.append(i[0])
mice_vec=np.unique(mice_pre)
#################

dic_time={}
dic_time['time_lock']='response' # response or stop_move
dic_time['start_window']=-2.0
dic_time['end_window']=0.2
dic_time['resol']=0.1
num_steps=int((dic_time['end_window']-dic_time['start_window'])/dic_time['resol'])
xx=np.arange(num_steps)*dic_time['resol']+dic_time['start_window']+dic_time['resol']/2.0

# Lick temporal resolution. No tocar!!
dt_lick={}
dt_lick['time_lock']='response' # response or stop_move
dt_lick['start_window']=0.0
dt_lick['end_window']=0.5
dt_lick['resol']=0.5
#
quantities={}
quant_vec_ct=np.array(['contacts','angle'])#'contacts'#,'tip_x','tip_y','fol_x','fol_y','angle'])
quant_vec_an=np.array([])#'angle','tip_x','tip_y','fol_x','fol_y'])
quant_vec_all=np.concatenate((quant_vec_ct,quant_vec_an))
quantities['contacts']=quant_vec_ct
quantities['analog']=quant_vec_an

features_slow=np.array([])

opto_dic=functions_miscellaneous.extract_opto(abs_path_pre)

###############################################

perf_pre=nan*np.zeros((2,len(mice_vec),num_steps))
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
      feat_prepre=functions_miscellaneous.create_feat(quantities=quantities,timings=timings,contact=contact,analog=analog,dic_time=dic_time)[index_lick][:,0]
      feat_pre=functions_miscellaneous.normalize_feat_time(feat_prepre)
      #feat_pre[np.isnan(feat_pre)]=0.0
      
      if i==0:
         reward_super=behavior['reward']
         stimulus_super=behavior['stimulus']
         position_super=behavior['position']
         choice_super=behavior['choice']
         feat_super=feat_pre.copy()
      else:
         position_super=np.concatenate((position_super,behavior['position']),axis=0)
         stimulus_super=np.concatenate((stimulus_super,behavior['stimulus']),axis=0)
         reward_super=np.concatenate((reward_super,behavior['reward']),axis=0)
         choice_super=np.concatenate((choice_super,behavior['choice']),axis=0)
         feat_super=np.concatenate((feat_super,feat_pre),axis=0)

   feat_norm=functions_miscellaneous.normalize_feat_time(feat_super)

   for j in range(num_steps):
      feat_resh=np.reshape(feat_norm[:,:,0:(j+1)],(len(feat_norm),-1))

      ####################################
      # Decode Stim
      perf_stim_reg_pre=nan*np.zeros((3,n_val,n_cv,n_decorr,len(reg_vec)))
      kf1=KFold(n_val)
      for g_val,index1 in enumerate(kf1.split(feat_norm)):
          train1=index1[0]
          val=index1[1]
          kf=KFold(n_cv)
          for g,index in enumerate(kf.split(feat_norm[train1])):
              train=index[0]
              test=index[1]
              values=nan*np.zeros(len(stimulus_super[train1][train]))
              values[(reward_super[train1][train]==-1)&(stimulus_super[train1][train]==-1)]=0
              values[(reward_super[train1][train]==-1)&(stimulus_super[train1][train]==1)]=1
              values[(reward_super[train1][train]==1)&(stimulus_super[train1][train]==-1)]=2
              values[(reward_super[train1][train]==1)&(stimulus_super[train1][train]==1)]=3
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
                      mod.fit(feat_resh[train1][train][index_t],stimulus_super[train1][train][index_t])
                      perf_stim_reg_pre[0,g_val,g,jj,jjj]=mod.score(feat_resh[train1][train],stimulus_super[train1][train])
                      perf_stim_reg_pre[1,g_val,g,jj,jjj]=mod.score(feat_resh[train1][test],stimulus_super[train1][test])
                      perf_stim_reg_pre[2,g_val,g,jj,jjj]=mod.score(feat_resh[val],stimulus_super[val])
      perf_stim_reg=np.nanmean(perf_stim_reg_pre,axis=(1,2,3))
      index_max_stim=np.where(perf_stim_reg[1]==np.nanmax(perf_stim_reg[1]))[0][0]
      perf_pre[0,k,j]=perf_stim_reg[2,index_max_stim]

      ######################################
      # Decode Choice
      perf_choi_reg_pre=nan*np.zeros((3,n_val,n_cv,n_decorr,len(reg_vec)))
      kf1=KFold(n_val)
      for g_val,index1 in enumerate(kf1.split(feat_norm)):
          train1=index1[0]
          val=index1[1]
          kf=KFold(n_cv)
          for g,index in enumerate(kf.split(feat_norm[train1])):
              train=index[0]
              test=index[1]
              values=nan*np.zeros(len(choice_super[train1][train]))
              values[(reward_super[train1][train]==-1)&(choice_super[train1][train]==-1)]=0
              values[(reward_super[train1][train]==-1)&(choice_super[train1][train]==1)]=1
              values[(reward_super[train1][train]==1)&(choice_super[train1][train]==-1)]=2
              values[(reward_super[train1][train]==1)&(choice_super[train1][train]==1)]=3
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
                      mod.fit(feat_resh[train1][train][index_t],choice_super[train1][train][index_t])
                      perf_choi_reg_pre[0,g_val,g,jj,jjj]=mod.score(feat_resh[train1][train],choice_super[train1][train])
                      perf_choi_reg_pre[1,g_val,g,jj,jjj]=mod.score(feat_resh[train1][test],choice_super[train1][test])
                      perf_choi_reg_pre[2,g_val,g,jj,jjj]=mod.score(feat_resh[val],choice_super[val])
      perf_choi_reg=np.nanmean(perf_choi_reg_pre,axis=(1,2,3))
      index_max_choi=np.where(perf_choi_reg[1]==np.nanmax(perf_choi_reg[1]))[0][0]
      perf_pre[1,k,j]=perf_choi_reg[2,index_max_choi]

   print (np.nanmean(perf_pre,axis=1))

perf_time_m=np.mean(perf_pre,axis=1)
perf_time_sem=sem(perf_pre,axis=1)
   

###############################
labels=['Stimulus','Choice']
colors=['green','blue']
fig=plt.figure(figsize=(2,2))
ax=fig.add_subplot(111)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.axvline(0,color='black',linestyle='--')
ax.plot(xx,0.5*np.ones(num_steps),color='black',linestyle='--')
ax.set_ylim([0.40,1.0])
for i in range(2):
   ax.plot(xx,perf_time_m[i],color=colors[i],label=labels[i])
   ax.fill_between(xx,perf_time_m[i]-perf_time_sem[i],perf_time_m[i]+perf_time_sem[i],color=colors[i],alpha=0.5)
plt.legend(loc='best')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Decoding Performance')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/DP_time_vali_ct_angle_%.2f.pdf'%dic_time['resol'],dpi=500,bbox_inches='tight')
