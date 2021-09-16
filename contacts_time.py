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
nan=float('nan')

task_type='discrimination'
# Discrimination: 10 mice
# Detection: 4 mice

var_split='stimulus'
norm=False

# Path Loads
abs_path_pre='/home/ramon/Dropbox/chris_randy/data/'
abs_path='/home/ramon/Dropbox/chris_randy/data/data_%s/sessions/'%task_type


################
# Extract Files
files_all=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_include_session_df')).loc[task_type]
files_good=files_all[(files_all['bad_perf']==False)]
files_vec_pre=files_good.index.values
files_vec=[]
#for i in range(len(files_vec_pre)):
   #if files_vec_pre[i][0]!='200CR':
   #   files_vec.append(files_vec_pre[i])
files_vec=np.array(files_vec_pre)
      
mice_pre=[]
for i in files_vec:
   mice_pre.append(i[0])
mice_vec=np.unique(mice_pre)
#################

dic_time={}
dic_time['time_lock']='response' # response or stop_move
dic_time['start_window']=-2.0
dic_time['end_window']=0.0
dic_time['resol']=0.25
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
quant_vec_ct=np.array(['contacts'])#'contacts'#,'tip_x','tip_y','fol_x','fol_y','angle'])                                                                                                                 
quant_vec_an=np.array([])#'angle','tip_x','tip_y','fol_x','fol_y'])
quant_vec_all=np.concatenate((quant_vec_ct,quant_vec_an))
quantities['contacts']=quant_vec_ct
quantities['analog']=quant_vec_an

features_slow=np.array([])

opto_dic=functions_miscellaneous.extract_opto(abs_path_pre)

###############################################
colors=['black','blue','green','red']

contacts_all=nan*np.zeros((3,len(mice_vec),4,num_steps))
contacts_sum=nan*np.zeros((3,len(mice_vec),4))
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
      feat_pre=functions_miscellaneous.create_feat(quantities=quantities,timings=timings,contact=contact,analog=analog,dic_time=dic_time)[index_lick][:,0]
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
         choice_super=np.concatenate((reward_super,behavior['choice']),axis=0)
         feat_super=np.concatenate((feat_super,feat_pre),axis=0)

   if var_split=='stimulus':
      labels=['Concave','Convex']
      variable=stimulus_super
   if var_split=='choice':
      variable=choice_super
   if var_split=='reward':
      labels=['Correct','Error']
      variable=reward_super
   index_c1=np.where(variable==1)[0]
   index_c2=np.where(variable==-1)[0]
      
   contacts_sum[0,k]=np.mean(np.sum(feat_super*dic_time['resol'],axis=2),axis=0)
   contacts_sum[1,k]=np.mean(np.sum(feat_super[index_c1]*dic_time['resol'],axis=2),axis=0)
   contacts_sum[2,k]=np.mean(np.sum(feat_super[index_c2]*dic_time['resol'],axis=2),axis=0)

   if norm==True:
      feat_super=functions_miscellaneous.normalize_feat_time(feat_super)
   if norm==False:
      None
   
   contacts_all[0,k]=np.nanmean(feat_super,axis=0)
   contacts_all[1,k]=np.nanmean(feat_super[index_c1],axis=0)
   contacts_all[2,k]=np.nanmean(feat_super[index_c2],axis=0)

   fig=plt.figure(figsize=(3,2))
   ax=fig.add_subplot(111)
   functions_miscellaneous.adjust_spines(ax,['left','bottom'])
   ax.axvline(0,color='black',linestyle='--')
   for i in range(4):
      if i==3:
         ax.plot(xx,contacts_all[1,k,i],color=colors[i],label=labels[0])
         ax.plot(xx,contacts_all[2,k,i],color=colors[i],label=labels[1],alpha=0.5)
      else:
         ax.plot(xx,contacts_all[1,k,i],color=colors[i])
         ax.plot(xx,contacts_all[2,k,i],color=colors[i],alpha=0.5)
   plt.legend(loc='best')
   ax.set_xlabel('Time (sec)')
   ax.set_ylabel('Contact rate (Hz)')
   #fig.savefig('/home/ramon/Dropbox/chris_randy/plots/contacts/contacts_time_%s_split_%s_norm_%s_mouse_%s.pdf'%(task_type,var_split,norm,mice_vec[k]),dpi=500,bbox_inches='tight')

contacts_all_m=np.nanmean(contacts_all,axis=1)
contacts_all_sem=sem(contacts_all,axis=1,nan_policy='omit')
contacts_all_diff_m=np.nanmean(contacts_all[1]-contacts_all[2],axis=1)
contacts_all_diff_sem=sem(contacts_all[1]-contacts_all[2],axis=1,nan_policy='omit')

###############################
labels=['C0','C1','C2','C3']
fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(111)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.axvline(0,color='black',linestyle='--')
for i in range(4):
   ax.plot(xx,contacts_all_m[0,i],color=colors[i],label=labels[i])
   ax.fill_between(xx,contacts_all_m[0,i]-contacts_all_sem[0,i],contacts_all_m[0,i]+contacts_all_sem[0,i],color=colors[i],alpha=0.5)
plt.legend(loc='best')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Contact rate (Hz)')
#fig.savefig('/home/ramon/Dropbox/chris_randy/plots/contacts/contacts_time_%s_norm_%s.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')

# Cond 1 vs Cond 2
labels=['Concave','Convex']
fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(111)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.axvline(0,color='black',linestyle='--')
for i in range(4):
   if i==3:
      ax.plot(xx,contacts_all_m[1,i],color=colors[i],label=labels[0])
      ax.plot(xx,contacts_all_m[2,i],color=colors[i],label=labels[1],alpha=0.5)
   else:
      ax.plot(xx,contacts_all_m[1,i],color=colors[i])
      ax.plot(xx,contacts_all_m[2,i],color=colors[i],alpha=0.5)
#plt.legend(loc='best')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Contact rate (Hz)')
#fig.savefig('/home/ramon/Dropbox/chris_randy/plots/contacts/contacts_time_%s_split_%s_norm_%s.pdf'%(task_type,var_split,norm),dpi=500,bbox_inches='tight')

###############################
# CUIDADO CON END TIME!!!
contacts_sum_diff_m=np.nanmean(contacts_sum[1]-contacts_sum[2],axis=0)
contacts_sum_diff_sem=sem(contacts_sum[1]-contacts_sum[2],axis=0,nan_policy='omit')

fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(111)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(6)-1,np.zeros(6),color='black',linestyle='--')
plt.xticks([0,1,2,3],['C0','C1','C2','C3'])
for i in range(4):
   ax.bar(i,contacts_sum_diff_m[i],yerr=contacts_sum_diff_sem[i],color=colors[i],alpha=0.75)
ax.set_ylabel(' Contact Difference\n Concave vs Convex')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/contacts/contact_diff_%s_norm_%s.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')
