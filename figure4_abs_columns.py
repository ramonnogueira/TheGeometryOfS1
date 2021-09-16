import torch
import numpy as np
import pickle as pkl
import nn_pytorch_encoders
import torch
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
import functions_miscellaneous
from scipy.stats import sem
import scipy
import pandas
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
nan=float('nan')
minf=float('-inf')
pinf=float('inf')

task_type='discrimination'
loss_type='gaussian'
type_trials=''
ablation='_full' 
model_vec=['FC_trivial','FC_one_deep_hundred_wide','FC_two_deep_hundred_wide','FC_three_deep_hundred_wide']
reg_vec=np.arange(20)
reg=1
n_cv=4
n_val=4
len_feat=318

v_used=np.array([[np.arange(5)+15,np.arange(5)+20],[np.arange(5)+15,np.arange(5)+25],[np.arange(5)+20,np.arange(5)+25]])
std_xor1=1e-3
n_sub=10
bar_vec=np.array(['C1','C2','C3'])

# Path Loads
abs_path_pre='/home/ramon/Dropbox/chris_randy/data/'
abs_path='/home/ramon/Dropbox/chris_randy/data/data_%s/sessions/'%task_type
abs_path_save='/home/ramon/Dropbox/chris_randy/data_analysis/%s/'%task_type

files_all=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_include_session_df')).loc[task_type]
files_neuro=pandas.read_pickle(os.path.join(abs_path_pre,'20200414_big_waveform_info_df'))
files_good=files_all[files_all['bad_perf']==False].index.values
files_vec=[]
for i in files_good:
   try:
      files_neuro.loc[i[1]]
      files_vec.append(i[1])
   except:
      None
files_vec=np.array(files_vec)
   
###############################################################
# Para recurrent hay que usar desde mas atras. Por ej desde -2.5                                                                                                                                         
dic_time={}
dic_time['time_lock']='response' # response or stop_move                                                                                                                                                 
dic_time['start_window']=-2.5
dic_time['end_window']=1.0
dic_time['size_kernel']=0.5
dic_time['resol']=0.1
num_steps=int((dic_time['end_window']-dic_time['start_window'])/dic_time['resol'])
n_kernel=int(dic_time['size_kernel']/dic_time['resol'])
xx=np.arange(num_steps)*dic_time['resol']+dic_time['start_window']+dic_time['resol']/2.0
n_steps_undo=(num_steps-int(dic_time['size_kernel']/dic_time['resol'])+1)

# Lick temporal resolution. No tocar!!
dt_lick={}
dt_lick['time_lock']='response' # response or stop_move
dt_lick['start_window']=0.0
dt_lick['end_window']=0.5
dt_lick['resol']=0.05

quantities={}
quant_vec_ct=np.array(['contacts','angle'])#,'tip_x','tip_y','fol_x','fol_y','angle'])
quant_vec_an=np.array(['angle'])#,'angle','tip_x','tip_y','fol_x','fol_y'])                                                                                                                              
quant_vec_all=np.concatenate((quant_vec_ct,quant_vec_an))
quantities['contacts']=quant_vec_ct
quantities['analog']=quant_vec_an
features_slow=np.array(['reward','stimulus','position','choice'])
 
# All
perf_surr_abs_col_pre=nan*np.zeros((len(files_vec),len(model_vec),n_val,n_cv,len(bar_vec),len(v_used),n_sub,2,2)) 
perf_surr_whisk_pre=nan*np.zeros((len(files_vec),len(model_vec),n_val,n_cv,len(bar_vec),len(v_used),n_sub,2,4)) 
for i in range(len(files_vec)):
   print (files_vec[i])
   apn=open(abs_path_save+'%s_%s_dictionary_feat_firing_%.2f_%.2f.pkl'%(files_vec[i],task_type,dic_time['size_kernel'],dic_time['resol']),'rb')
   datos=pkl.load(apn)
   apn.close()
   feat_pre=datos['feat_pre']
   firing_rate=datos['firing_rate']
   lick_rate=datos['lick_rate']
   behavior=datos['behavior']
   num_neu=len(firing_rate[0])

   # Column Location
   location=functions_miscellaneous.extract_electrode_location(np.array([files_vec[i]]),abs_path_pre,'20191007_electrode_locations.csv')['location_c']
   index_col=(location[0]==bar_vec)
   print (location[0],len(location))
   if np.sum(index_col)>0:
      # Put features in a nice format pre-encoding
      encoding=functions_miscellaneous.crear_encoding_general(features_fast=feat_pre[:,0],features_slow=features_slow,firing_rate=firing_rate,lick_rate=lick_rate,behavior=behavior)
      reward_encoding_pre=encoding['reward']
      feat_norm=functions_miscellaneous.normalize_feat_time_general(encoding['features'])
      if loss_type=='gaussian':
         firing_norm=functions_miscellaneous.normalize_fr_time_general(fr=encoding['firing_rate'])
      if loss_type=='poisson':
         firing_norm=(encoding['firing_rate']*dic_time['resol'])
   
      # LOAD MODELS
      for ii in range(len(model_vec)):
         print (model_vec[ii])
         load_models=functions_miscellaneous.load_models(abs_path_save=abs_path_save,loss_type=loss_type,arx=files_vec[i],mod_arch=model_vec[ii],type_trials=type_trials,ablation=ablation,reg_vec=reg_vec,len_feat=len_feat,num_neu=num_neu)
         arx_models=load_models['arx_models']
         model=load_models['model']
         index_cv=load_models['index_cv']
         index_max_lr=load_models['index_max_lr']
       
         for j in range(n_val):
            index_val=index_cv['val_%i'%j]
            for jj in range(n_cv):
               model.load_state_dict(arx_models['g_val_%i_lr_%i_cv_%i'%(j,index_max_lr,jj)])
               model.eval()

               encoding_all=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=firing_norm[index_val],features_general=feat_norm[index_val],features_slow=features_slow,dic_time=dic_time,reward=reward_encoding_pre[index_val])
               encoding_nonorm=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=encoding['firing_rate'][index_val],features_general=encoding['features'][index_val],features_slow=features_slow,dic_time=dic_time,reward=reward_encoding_pre[index_val])
               feat_encoding=encoding_all['features']
               feat_encoding_nonorm=encoding_nonorm['features']
            
               for h in range(len(v_used)):
                  index_used1=np.concatenate((v_used[h,0],v_used[h,1]))
                  feat_sum=np.sum(feat_encoding_nonorm[:,index_used1],axis=1)
                  index_ct=(feat_sum!=0)
                
                  create_xor=functions_miscellaneous.create_xor_median_encoding_new(feat=feat_encoding[index_ct],v_used=v_used[h],std=std_xor1)
                  feat_noise_all=create_xor['feat_noise_all']
                  feat_binary_orig=create_xor['feat_binary_orig'] 
                  xor=create_xor['xor']
                  lin=create_xor['lin']

                  feat_all=torch.zeros((len(feat_noise_all),len_feat))
                  feat_all_var=Variable(torch.from_numpy(np.array(feat_noise_all,dtype=np.float32)),requires_grad=False)
                  feat_all[:,v_used[h]]=feat_all_var[:,v_used[h]]
                  fire_all=model(feat_all).detach().numpy()

                  for gg in range(n_sub):
                     neu_min=10
                     index_neu_sub=np.random.choice(np.arange(num_neu),neu_min,replace=False)
                     abstract=functions_miscellaneous.abstraction_2D(fire_all[:,index_neu_sub],feat_binary_orig)
                     perf_surr_abs_col_pre[i,ii,j,jj,index_col,h,gg]=abstract[:,:,1]

                     for kk in range(len(v_used[h])):
                        ggg=-1
                        skf=StratifiedKFold(n_splits=4)
                        for train, test in skf.split(fire_all,feat_binary_orig[:,kk]):
                           ggg=(ggg+1)
                           supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
                           mod=supp.fit(fire_all[train],feat_binary_orig[train][:,kk])
                           perf_surr_whisk_pre[i,ii,j,jj,index_col,h,gg,kk,ggg]=supp.score(fire_all[test],feat_binary_orig[test][:,kk])

   print ('abs cols  ',np.nanmean(perf_surr_abs_col_pre,axis=(0,2,3,6,8))[1])
   print ('whisk  ',np.nanmean(perf_surr_whisk_pre,axis=(0,2,3,6,8))[1])

perf_abs_ind=np.nanmean(perf_surr_abs_col_pre,axis=(2,3,6,8))
perf_abs_c1=0.5*(perf_abs_ind[:,:,:,0,0]+perf_abs_ind[:,:,:,1,0])
perf_abs_c2=0.5*(perf_abs_ind[:,:,:,0,1]+perf_abs_ind[:,:,:,2,0])
perf_abs_c3=0.5*(perf_abs_ind[:,:,:,1,1]+perf_abs_ind[:,:,:,2,1])
perf_abs_c1_m=np.nanmean(perf_abs_c1,axis=0)
perf_abs_c1_sem=sem(perf_abs_c1,axis=0,nan_policy='omit')
perf_abs_c2_m=np.nanmean(perf_abs_c2,axis=0)
perf_abs_c2_sem=sem(perf_abs_c2,axis=0,nan_policy='omit')
perf_abs_c3_m=np.nanmean(perf_abs_c3,axis=0)
perf_abs_c3_sem=sem(perf_abs_c3,axis=0,nan_policy='omit')

perf_wsk_ind=np.nanmean(perf_surr_whisk_pre,axis=(2,3,6,8))
perf_wsk_c1=0.5*(perf_wsk_ind[:,:,:,0,0]+perf_wsk_ind[:,:,:,1,0])
perf_wsk_c2=0.5*(perf_wsk_ind[:,:,:,0,1]+perf_wsk_ind[:,:,:,2,0])
perf_wsk_c3=0.5*(perf_wsk_ind[:,:,:,1,1]+perf_wsk_ind[:,:,:,2,1])
perf_wsk_c1_m=np.nanmean(perf_wsk_c1,axis=0)
perf_wsk_c1_sem=sem(perf_wsk_c1,axis=0,nan_policy='omit')
perf_wsk_c2_m=np.nanmean(perf_wsk_c2,axis=0)
perf_wsk_c2_sem=sem(perf_wsk_c2,axis=0,nan_policy='omit')
perf_wsk_c3_m=np.nanmean(perf_wsk_c3,axis=0)
perf_wsk_c3_sem=sem(perf_wsk_c3,axis=0,nan_policy='omit')

#######################################################
# ALL

fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(1,1,1)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.set_ylim([0.4,1.0])
ax.set_ylabel('Decoding Performance')
width=0.25
alpha=0.75
col_vec=['blue','green','red']
for i in range(3):
    ax.bar(-width+i,perf_abs_c1_m[1,i],yerr=perf_abs_c1_sem[1,i],width=width,color='blue',alpha=alpha)
    ax.bar(i,perf_abs_c2_m[1,i],yerr=perf_abs_c2_sem[1,i],width=width,color='green',alpha=alpha)
    ax.bar(width+i,perf_abs_c3_m[1,i],yerr=perf_abs_c3_sem[1,i],width=width,color='red',alpha=alpha)
ax.plot(np.linspace(-0.5,2.5,10),0.5*np.ones(10),color='black',linestyle='--')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure4_abs_cols_%.3f_new.pdf'%(std_xor1),dpi=500,bbox_inches='tight')

# Whiskers columns
fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(1,1,1)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.set_ylim([0.4,1.0])
ax.set_ylabel('Decoding Performance')
width=0.25
alpha=0.75
col_vec=['blue','green','red']
for i in range(3):
    ax.bar(-width+i,perf_wsk_c1_m[1,i],yerr=perf_wsk_c1_sem[1,i],width=width,color='blue',alpha=alpha)
    ax.bar(i,perf_wsk_c2_m[1,i],yerr=perf_wsk_c2_sem[1,i],width=width,color='green',alpha=alpha)
    ax.bar(width+i,perf_wsk_c3_m[1,i],yerr=perf_wsk_c3_sem[1,i],width=width,color='red',alpha=alpha)
ax.plot(np.linspace(-0.5,2.5,10),0.5*np.ones(10),color='black',linestyle='--')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure4_wsk_cols_%.3f_new.pdf'%(std_xor1),dpi=500,bbox_inches='tight')


         
