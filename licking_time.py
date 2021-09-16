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

norm=False

# Path Loads
abs_path_pre='/home/ramon/Dropbox/chris_randy/data/'
abs_path='/home/ramon/Dropbox/chris_randy/data/data_%s/sessions/'%task_type

# Path save
abs_path_save='/home/ramon/Dropbox/chris_randy/plots/licks/'

################
# Extract Files
files_all=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_include_session_df')).loc[task_type]
files_good=files_all[(files_all['bad_perf']==False)]
files_vec_pre=files_good.index.values
files_vec=[]
#for i in range(len(files_vec_pre)):
#   if files_vec_pre[i][0]!='200CR':
#      files_vec.append(files_vec_pre[i])
files_vec=np.array(files_vec_pre)
      
mice_pre=[]
for i in files_vec:
   mice_pre.append(i[0])
mice_vec=np.unique(mice_pre)
#################

dic_time={}
dic_time['time_lock']='response' # response or stop_move
dic_time['start_window']=-2.0
dic_time['end_window']=0.5
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

features_slow=np.array([])
###############################################

opto_dic=functions_miscellaneous.extract_opto(abs_path_pre)
lick_mouse=nan*np.zeros((3,3,len(mice_vec),len(xx)))
perf_mouse=nan*np.zeros((len(mice_vec),len(xx)))
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
      lick_norm_rate=functions_miscellaneous.normalize_lick_time(lick_rate)
         
      behavior=functions_miscellaneous.extract_behavior(index_use,index_lick,abs_path+files_mouse[i])
      
      if i==0:
         lick_super=lick_rate.copy()
         stimulus_super=behavior['stimulus']
         choice_super=behavior['choice']
         reward_super=behavior['reward']
      else:
         lick_super=np.concatenate((lick_super,lick_rate),axis=0)
         stimulus_super=np.concatenate((stimulus_super,behavior['stimulus']),axis=0)
         choice_super=np.concatenate((choice_super,behavior['choice']),axis=0)
         reward_super=np.concatenate((reward_super,behavior['reward']),axis=0)

   index_correct=np.where(reward_super==1)[0]
   index_incorrect=np.where(reward_super==-1)[0]

   # All trials
   perf_mouse[k]=len(reward_super[reward_super==1])/float(len(reward_super))*np.ones(len(xx))
   if norm==True:
      lick_mouse[0,0,k]=np.mean(functions_miscellaneous.normalize_lick_time(abs(lick_super)),axis=0)
   if norm==False:
      lick_mouse[0,0,k]=np.mean(abs(lick_super),axis=0)
   lick_mouse[0,1,k]=functions_miscellaneous.prob_quantity(lick_rate=lick_super,quantity=stimulus_super)
   lick_mouse[0,2,k]=functions_miscellaneous.prob_quantity(lick_rate=lick_super,quantity=choice_super)

   # Correct trials
   if norm==True:
      lick_mouse[1,0,k]=np.mean(functions_miscellaneous.normalize_lick_time(abs(lick_super[index_correct])),axis=0)
   if norm==False:
      lick_mouse[1,0,k]=np.mean(abs(lick_super[index_correct]),axis=0)
   lick_mouse[1,1,k]=functions_miscellaneous.prob_quantity(lick_rate=lick_super[index_correct],quantity=stimulus_super[index_correct])
   lick_mouse[1,2,k]=functions_miscellaneous.prob_quantity(lick_rate=lick_super[index_correct],quantity=choice_super[index_correct])
   # Incorrect trials
   if norm==True:
      lick_mouse[2,0,k]=np.mean(functions_miscellaneous.normalize_lick_time(abs(lick_super[index_incorrect])),axis=0)
   if norm==False:
      lick_mouse[2,0,k]=np.mean(abs(lick_super[index_incorrect]),axis=0)
   lick_mouse[2,1,k]=functions_miscellaneous.prob_quantity(lick_rate=lick_super[index_incorrect],quantity=stimulus_super[index_incorrect])
   lick_mouse[2,2,k]=functions_miscellaneous.prob_quantity(lick_rate=lick_super[index_incorrect],quantity=choice_super[index_incorrect])

lick_all_mean=np.nanmean(lick_mouse,axis=2)
lick_all_sem=sem(lick_mouse,axis=2,nan_policy='omit')
perf_mean=np.nanmean(perf_mouse,axis=0)
perf_sem=sem(perf_mouse,axis=0,nan_policy='omit')

##################################
# All trials
fig1=plt.figure(figsize=(3,2))
ax1=fig1.add_subplot(111)
functions_miscellaneous.adjust_spines(ax1,['left','bottom'])
ax1.set_xticks([-1.5,-1.0,-0.5,0])
ax1.axvline(0,color='black',linestyle='--')
ax1.plot(xx,lick_all_mean[0,0],color='black')
ax1.fill_between(xx,lick_all_mean[0,0]-lick_all_sem[0,0],lick_all_mean[0,0]+lick_all_sem[0,0],color=(0,0,0,0.5))
ax1.set_xlabel('Time (sec)')
if norm==False:
   ax1.set_ylabel('Lick rate (Hz)')
if norm==True:
   ax1.set_ylabel('Norm. Lick rate (Hz)')
#fig1.savefig(abs_path_save+'lick_rate_all_%s_norm_%s_all_trials.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')

fig2=plt.figure(figsize=(3,2))
ax2=fig2.add_subplot(111)
functions_miscellaneous.adjust_spines(ax2,['left','bottom'])
ax2.set_xticks([-1.5,-1.0,-0.5,0])
ax2.axvline(0,color='black',linestyle='--')
ax2.plot(xx,lick_all_mean[0,1],color='black')
ax2.fill_between(xx,lick_all_mean[0,1]-lick_all_sem[0,1],lick_all_mean[0,1]+lick_all_sem[0,1],color=(0,0,0,0.5))
ax2.plot(xx,perf_mean,color='brown')
ax2.fill_between(xx,perf_mean-perf_sem,perf_mean+perf_sem,color='goldenrod')
ax2.plot(xx,0.5*np.ones(len(xx)),color='black')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Prob(Lick Correct Side)')
#fig2.savefig(abs_path_save+'prob_correct_lick_%s_norm_%s_all_trials.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')

fig3=plt.figure(figsize=(3,2))
ax3=fig3.add_subplot(111)
functions_miscellaneous.adjust_spines(ax3,['left','bottom'])
ax3.set_xticks([-1.5,-1.0,-0.5,0])
ax3.axvline(0,color='black',linestyle='--')
ax3.plot(xx,lick_all_mean[0,2],color='black')
ax3.fill_between(xx,lick_all_mean[0,2]-lick_all_sem[0,2],lick_all_mean[0,2]+lick_all_sem[0,2],color=(0,0,0,0.5))
ax3.plot(xx,0.5*np.ones(len(xx)),color='black')
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel('Prob(Lick=Decision|Lick)')
#fig3.savefig(abs_path_save+'prob_lick_final_dec_all_%s_norm_%s_all_trials.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')

##################################
# Correct trials and Incorrect
fig1=plt.figure(figsize=(3,2))
ax1=fig1.add_subplot(111)
functions_miscellaneous.adjust_spines(ax1,['left','bottom'])
ax1.set_xticks([-1.5,-1.0,-0.5,0])
ax1.axvline(0,color='black',linestyle='--')
ax1.plot(xx,lick_all_mean[1,0],color='black')
ax1.fill_between(xx,lick_all_mean[1,0]-lick_all_sem[1,0],lick_all_mean[1,0]+lick_all_sem[1,0],color=(0,0,0,0.5))
ax1.plot(xx,lick_all_mean[2,0],color='red')
ax1.fill_between(xx,lick_all_mean[2,0]-lick_all_sem[2,0],lick_all_mean[2,0]+lick_all_sem[2,0],color=(1,0,0,0.5))
ax1.set_xlabel('Time (sec)')
if norm==False:
   ax1.set_ylabel('Lick rate (Hz)')
if norm==True:
   ax1.set_ylabel('Norm. Lick rate (Hz)')
#fig1.savefig(abs_path_save+'lick_rate_all_%s_corr_incorr_norm_%s.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')

fig2=plt.figure(figsize=(3,2))
ax2=fig2.add_subplot(111)
functions_miscellaneous.adjust_spines(ax2,['left','bottom'])
ax2.set_xticks([-1.5,-1.0,-0.5,0])
ax2.axvline(0,color='black',linestyle='--')
ax2.plot(xx,lick_all_mean[1,1],color='black')
ax2.fill_between(xx,lick_all_mean[1,1]-lick_all_sem[1,1],lick_all_mean[1,1]+lick_all_sem[1,1],color=(0,0,0,0.5))
ax2.plot(xx,lick_all_mean[2,1],color='red')
ax2.fill_between(xx,lick_all_mean[2,1]-lick_all_sem[2,1],lick_all_mean[2,1]+lick_all_sem[2,1],color=(1,0,0,0.5))
ax2.plot(xx,0.5*np.ones(len(xx)),color='black')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Prob(Lick Correct Side)')
#fig2.savefig(abs_path_save+'prob_correct_lick_%s_corr_incorr_norm_%s.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')

fig3=plt.figure(figsize=(3,2))
ax3=fig3.add_subplot(111)
functions_miscellaneous.adjust_spines(ax3,['left','bottom'])
ax3.set_xticks([-1.5,-1.0,-0.5,0])
ax3.axvline(0,color='black',linestyle='--')
ax3.plot(xx,lick_all_mean[1,2],color='black')
ax3.fill_between(xx,lick_all_mean[1,2]-lick_all_sem[1,2],lick_all_mean[1,2]+lick_all_sem[1,2],color=(0,0,0,0.5))
ax3.plot(xx,lick_all_mean[2,2],color='red')
ax3.fill_between(xx,lick_all_mean[2,2]-lick_all_sem[2,2],lick_all_mean[2,2]+lick_all_sem[2,2],color=(1,0,0,0.5))
ax3.plot(xx,0.5*np.ones(len(xx)),color='black')
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel('Prob(Lick=Decision|Lick)')
#fig3.savefig(abs_path_save+'prob_lick_final_dec_all_%s_corr_incorr_norm_%s.pdf'%(task_type,norm),dpi=500,bbox_inches='tight')

