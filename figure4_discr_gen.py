import numpy as np
import pickle as pkl
import nn_pytorch_encoders
import torch
from torch.nn import Linear
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from scipy.stats import sem
import functions_miscellaneous
from scipy.stats import sem
from scipy.stats import pearsonr
import scipy
import pandas
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
nan=float('nan')
minf=float('-inf')
pinf=float('inf')

def yx(x,wei,inter):
   t1=wei[0]*x
   t2=inter*np.ones(np.shape(x))
   y=(-t1-t2)/wei[1]
   return y
########################################################

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

#v_used=np.array([[np.arange(5)+15,np.arange(5)+20],[np.arange(5)+15,np.arange(5)+25],[np.arange(5)+20,np.arange(5)+25]])
#v_used=np.array([[np.arange(5)+55,np.arange(5)+60],[np.arange(5)+55,np.arange(5)+65],[np.arange(5)+60,np.arange(5)+65]])
v_used=np.array([[np.arange(5)+15,np.arange(5)+55],
                 [np.arange(5)+15,np.arange(5)+60],
                 [np.arange(5)+15,np.arange(5)+65],
                 [np.arange(5)+20,np.arange(5)+55],
                 [np.arange(5)+20,np.arange(5)+60],
                 [np.arange(5)+20,np.arange(5)+65],
                 [np.arange(5)+25,np.arange(5)+55],
                 [np.arange(5)+25,np.arange(5)+60],
                 [np.arange(5)+25,np.arange(5)+65]]) #C1 A1; C1 A2; C1 A3; C2 A1; C2 A2; C2 A3; C3 A1; C3 A2; C3 A3

neu_vec=np.array([2,10,100,1000,10000])
std_xor1=1e-3
print ('std xor',std_xor1)

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

num_neurons=nan*np.zeros(len(files_vec))
perf_lin_pre=nan*np.zeros((len(files_vec),len(model_vec),n_val,len(v_used),n_cv,len(neu_vec)+1,4)) 
perf_xor_pre=nan*np.zeros((len(files_vec),len(model_vec),n_val,len(v_used),n_cv,len(neu_vec)+1,4))
perf_abs_pre=nan*np.zeros((len(files_vec),len(model_vec),n_val,len(v_used),n_cv,len(neu_vec)+1,2,2))
perf_wsk_pre=nan*np.zeros((len(files_vec),len(model_vec),n_val,len(v_used),n_cv,len(neu_vec)+1,2,4)) 
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
    num_neurons[i]=num_neu

    # Put features in a nice format pre-encoding
    encoding=functions_miscellaneous.crear_encoding_general(features_fast=feat_pre[:,0],features_slow=features_slow,firing_rate=firing_rate,lick_rate=lick_rate,behavior=behavior)
    reward_encoding_pre=encoding['reward']
    feat_norm=functions_miscellaneous.normalize_feat_time_general(encoding['features'])
    if loss_type=='gaussian':
        firing_norm=functions_miscellaneous.normalize_fr_time_general(fr=encoding['firing_rate'])
    if loss_type=='poisson':
        firing_norm=(encoding['firing_rate']*dic_time['resol'])

    # Loop over encoding models
    for ii in range(len(model_vec)):
        print (model_vec[ii])
        load_models=functions_miscellaneous.load_models(abs_path_save=abs_path_save,loss_type=loss_type,arx=files_vec[i],mod_arch=model_vec[ii],type_trials=type_trials,ablation=ablation,reg_vec=reg_vec,len_feat=len_feat,num_neu=num_neu)
        arx_orig=load_models['arx_models']
        model_orig=load_models['model']
        index_cv=load_models['index_cv']
        index_max_lr=load_models['index_max_lr']
        
        for j in range(n_val):
            #print (j)
            index_val=index_cv['val_%i'%j]
            encoding_all=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=firing_norm[index_val],features_general=feat_norm[index_val],features_slow=features_slow,dic_time=dic_time,reward=reward_encoding_pre[index_val])
            encoding_nonorm=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=encoding['firing_rate'][index_val],features_general=encoding['features'][index_val],features_slow=features_slow,dic_time=dic_time,reward=reward_encoding_pre[index_val])
            feat_encoding=encoding_all['features']
            feat_encoding_nonorm=encoding_nonorm['features']

            for h in range(len(v_used)):
                #print ('  ',h)
                index_used1=np.concatenate((v_used[h,0],v_used[h,1]))
                #feat_sum=np.sum(feat_encoding_nonorm[:,index_used1],axis=1)
                feat_sum=np.sum(feat_encoding_nonorm[:,v_used[h,0]],axis=1) # Cuidado!!
                index_ct=(feat_sum!=0)
                
                try:    
                    create_xor=functions_miscellaneous.create_xor_median_encoding_new(feat=feat_encoding[index_ct],v_used=v_used[h],std=std_xor1)
                    feat_noise_all=create_xor['feat_noise_all']
                    feat_noise=create_xor['feat_noise']
                    feat_binary_orig=create_xor['feat_binary_orig'] 
                    xor=create_xor['xor']
                    lin=create_xor['lin']               

                    for jj in range(n_cv):
                        #print ('    ',jj)
                        model_orig.load_state_dict(arx_orig['g_val_%i_lr_%i_cv_%i'%(j,index_max_lr,jj)])
                        feat_all=torch.zeros((len(feat_noise_all),len_feat))
                        feat_all_var=Variable(torch.from_numpy(np.array(feat_noise_all,dtype=np.float32)),requires_grad=False)
                        feat_all[:,index_used1]=feat_all_var[:,index_used1]
                        fire_all=model_orig(feat_all).detach().numpy()
                        # Linear
                        skf=StratifiedKFold(n_splits=4)
                        ggg=-1
                        for train, test in skf.split(fire_all,lin):
                            ggg=(ggg+1)
                            supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
                            mod=supp.fit(fire_all[train],lin[train])
                            perf_lin_pre[i,ii,j,h,jj,0,ggg]=supp.score(fire_all[test],lin[test])  
                        # XOR
                        skf=StratifiedKFold(n_splits=4)
                        ggg=-1
                        for train, test in skf.split(fire_all,xor):
                            ggg=(ggg+1)
                            supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
                            mod=supp.fit(fire_all[train],xor[train])
                            perf_xor_pre[i,ii,j,h,jj,0,ggg]=supp.score(fire_all[test],xor[test])
                        # Abstraction
                        abstract=functions_miscellaneous.abstraction_2D(fire_all,feat_binary_orig,reg=reg)
                        perf_abs_pre[i,ii,j,h,jj,0]=abstract[:,:,1]
                        # Whiskers
                        for kk in range(len(v_used[h])):
                            skf=StratifiedKFold(n_splits=4)
                            ggg=-1
                            for train, test in skf.split(fire_all,feat_binary_orig[:,kk]):
                                ggg=(ggg+1)
                                supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
                                mod=supp.fit(fire_all[train],feat_binary_orig[train][:,kk])
                                perf_wsk_pre[i,ii,j,h,jj,0,kk,ggg]=supp.score(fire_all[test],feat_binary_orig[test][:,kk])
                                
                        #Loop over Population Size
                        for hh in range(len(neu_vec)):
                            #print ('    ',hh)
                            load_models2=functions_miscellaneous.load_models(abs_path_save=abs_path_save,loss_type=loss_type,arx=files_vec[i],mod_arch=model_vec[ii],type_trials=type_trials,ablation=ablation,reg_vec=reg_vec,len_feat=len_feat,num_neu=num_neu)
                            arx=load_models2['arx_models']
                            model=load_models2['model']
                            model.load_state_dict(arx['g_val_%i_lr_%i_cv_%i'%(j,index_max_lr,jj)])
                            if model_vec[ii]=='FC_trivial':
                                wei_new,bias_new=functions_miscellaneous.expand_weights(neu_vec[hh],model.linear1.weight.data,model.linear1.bias.data)
                                model.linear1.weight=torch.nn.Parameter(wei_new)
                                model.linear1.bias=torch.nn.Parameter(bias_new)
                            if model_vec[ii]=='FC_one_deep_hundred_wide':
                                wei_new,bias_new=functions_miscellaneous.expand_weights(neu_vec[hh],model.linear2.weight.data,model.linear2.bias.data)
                                model.linear2.weight=torch.nn.Parameter(wei_new)
                                model.linear2.bias=torch.nn.Parameter(bias_new)
                            if model_vec[ii]=='FC_two_deep_hundred_wide':
                                wei_new,bias_new=functions_miscellaneous.expand_weights(neu_vec[hh],model.linear3.weight.data,model.linear3.bias.data)
                                model.linear3.weight=torch.nn.Parameter(wei_new)
                                model.linear3.bias=torch.nn.Parameter(bias_new)
                            if model_vec[ii]=='FC_three_deep_hundred_wide':
                                wei_new,bias_new=functions_miscellaneous.expand_weights(neu_vec[hh],model.linear4.weight.data,model.linear4.bias.data)
                                model.linear4.weight=torch.nn.Parameter(wei_new)
                                model.linear4.bias=torch.nn.Parameter(bias_new)
                            model.eval()
                            fire_all=model(feat_all).detach().numpy()
                            
                            # Linear
                            skf=StratifiedKFold(n_splits=4)
                            ggg=-1
                            for train, test in skf.split(fire_all,lin):
                                ggg=(ggg+1)
                                supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
                                mod=supp.fit(fire_all[train],lin[train])
                                perf_lin_pre[i,ii,j,h,jj,hh+1,ggg]=supp.score(fire_all[test],lin[test])
                            # XOR
                            skf=StratifiedKFold(n_splits=4)
                            ggg=-1
                            for train, test in skf.split(fire_all,xor):
                                ggg=(ggg+1)
                                supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
                                mod=supp.fit(fire_all[train],xor[train])
                                perf_xor_pre[i,ii,j,h,jj,hh+1,ggg]=supp.score(fire_all[test],xor[test])
                            # Abstraction
                            abstract=functions_miscellaneous.abstraction_2D(fire_all,feat_binary_orig,reg=reg)
                            perf_abs_pre[i,ii,j,h,jj,hh+1]=abstract[:,:,1]
                            # Whiskers
                            for kk in range(2):
                                skf=StratifiedKFold(n_splits=4)
                                ggg=-1
                                for train, test in skf.split(fire_all,feat_binary_orig[:,kk]):
                                    ggg=(ggg+1)
                                    supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
                                    mod=supp.fit(fire_all[train],feat_binary_orig[train][:,kk])
                                    perf_wsk_pre[i,ii,j,h,jj,hh+1,kk,ggg]=supp.score(fire_all[test],feat_binary_orig[test][:,kk])
                except:
                    print ('Aqui')
                                
    print ('lin   ',np.nanmean(perf_lin_pre,axis=(0,2,3,4,6)))
    print ('xor   ',np.nanmean(perf_xor_pre,axis=(0,2,3,4,6)))
    print ('abs   ',np.nanmean(perf_abs_pre,axis=(0,2,4,5,7)))
    print ('wsk   ',np.nanmean(perf_wsk_pre,axis=(0,2,4,5,7)))
    
num_neurons_m=np.nanmean(num_neurons)
perf_lin=np.nanmean(perf_lin_pre,axis=(2,3,4,6))
perf_xor=np.nanmean(perf_xor_pre,axis=(2,3,4,6))
perf_abs=np.nanmean(perf_abs_pre,axis=(2,3,4,6,7))
perf_wsk=np.nanmean(perf_wsk_pre,axis=(2,3,4,6,7))
perf_abs_ind=np.nanmean(perf_abs_pre,axis=(2,4,7))
perf_abs_c1=0.5*(perf_abs_ind[:,:,0,:,0]+perf_abs_ind[:,:,1,:,0])
perf_abs_c2=0.5*(perf_abs_ind[:,:,0,:,1]+perf_abs_ind[:,:,2,:,0])
perf_abs_c3=0.5*(perf_abs_ind[:,:,1,:,1]+perf_abs_ind[:,:,2,:,1])
perf_wsk_ind=np.nanmean(perf_wsk_pre,axis=(2,4,7))
perf_wsk_c1=0.5*(perf_wsk_ind[:,:,0,:,0]+perf_wsk_ind[:,:,1,:,0])
perf_wsk_c2=0.5*(perf_wsk_ind[:,:,0,:,1]+perf_wsk_ind[:,:,2,:,0])
perf_wsk_c3=0.5*(perf_wsk_ind[:,:,1,:,1]+perf_wsk_ind[:,:,2,:,1])

perf_lin_m=np.nanmean(perf_lin,axis=0)
perf_lin_sem=sem(perf_lin,axis=0,nan_policy='omit')
perf_xor_m=np.nanmean(perf_xor,axis=0)
perf_xor_sem=sem(perf_xor,axis=0,nan_policy='omit')
perf_abs_m=np.nanmean(perf_abs,axis=0)
perf_abs_sem=sem(perf_abs,axis=0,nan_policy='omit')
perf_wsk_m=np.nanmean(perf_wsk,axis=0)
perf_wsk_sem=sem(perf_wsk,axis=0,nan_policy='omit')

perf_abs_c1_m=np.nanmean(perf_abs_c1,axis=0)
perf_abs_c1_sem=sem(perf_abs_c1,axis=0,nan_policy='omit')
perf_abs_c2_m=np.nanmean(perf_abs_c2,axis=0)
perf_abs_c2_sem=sem(perf_abs_c2,axis=0,nan_policy='omit')
perf_abs_c3_m=np.nanmean(perf_abs_c3,axis=0)
perf_abs_c3_sem=sem(perf_abs_c3,axis=0,nan_policy='omit')

perf_wsk_c1_m=np.nanmean(perf_wsk_c1,axis=0)
perf_wsk_c1_sem=sem(perf_wsk_c1,axis=0,nan_policy='omit')
perf_wsk_c2_m=np.nanmean(perf_wsk_c2,axis=0)
perf_wsk_c2_sem=sem(perf_wsk_c2,axis=0,nan_policy='omit')
perf_wsk_c3_m=np.nanmean(perf_wsk_c3,axis=0)
perf_wsk_c3_sem=sem(perf_wsk_c3,axis=0,nan_policy='omit')


#############################################################################################
# Performance vs Neurons

alpha_vec=[0.2,0.5,0.8,1.0]

# Linear Task
fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(1,1,1)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.set_ylim([0.4,1.0])
ax.set_ylabel('Decoding Performance')
ax.set_xscale('log')
ax.set_xlabel('Number of Neurons')
for i in range(len(model_vec)):
    ax.errorbar(num_neurons_m,perf_lin_m[i,0],yerr=perf_lin_sem[i,0],color='black',alpha=alpha_vec[i])
    ax.errorbar(neu_vec,perf_lin_m[i,1:],yerr=perf_lin_sem[i,1:],color='green',alpha=alpha_vec[i])
ax.plot(neu_vec,0.5*np.ones(len(neu_vec)),color='black',linestyle='--')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure_virtualtask2D_linear_vs_N_angle_ct.pdf',dpi=500,bbox_inches='tight')

# XOR Task
fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(1,1,1)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.set_ylim([0.4,1.0])
ax.set_ylabel('Decoding Performance')
ax.set_xscale('log')
ax.set_xlabel('Number of Neurons')
for i in range(len(model_vec)):
    ax.errorbar(num_neurons_m,perf_xor_m[i,0],yerr=perf_xor_sem[i,0],color='black',alpha=alpha_vec[i])
    ax.errorbar(neu_vec,perf_xor_m[i,1:],yerr=perf_xor_sem[i,1:],color='green',alpha=alpha_vec[i])
ax.plot(neu_vec,0.5*np.ones(len(neu_vec)),color='black',linestyle='--')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure_virtualtask2D_xor_vs_N_angle_ct.pdf',dpi=500,bbox_inches='tight')

# Abstraction Task
fig=plt.figure(figsize=(3,2))
ax=fig.add_subplot(1,1,1)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.set_ylim([0.4,1.0])
ax.set_ylabel('Decoding Performance')
ax.set_xscale('log')
ax.set_xlabel('Number of Neurons')
for i in range(len(model_vec)):
    ax.errorbar(num_neurons_m,perf_abs_m[i,0],yerr=perf_abs_sem[i,0],color='black',alpha=alpha_vec[i])
    ax.errorbar(neu_vec,perf_abs_m[i,1:],yerr=perf_abs_sem[i,1:],color='green',alpha=alpha_vec[i])
ax.plot(neu_vec,0.5*np.ones(len(neu_vec)),color='black',linestyle='--')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure_virtualtask2D_abs_vs_N_angle_ct.pdf',dpi=500,bbox_inches='tight')


#######################################################
# Bar plots

width=0.2
alpha_vec=[0.4,0.6,0.8,1.0]
for j in range(len(neu_vec)+1):
    # Linear task
    fig=plt.figure(figsize=(1.5,1.8))
    ax=fig.add_subplot(1,1,1)
    functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.set_ylim([0.4,1.0])
    ax.set_ylabel('Decoding Performance')
    for i in range(len(model_vec)):
        ax.bar(-1.5*width+width*i,perf_lin_m[i,j],yerr=perf_lin_sem[i,j],width=width,color='green',alpha=alpha_vec[i])
    ax.plot(np.linspace(-0.5,0.5,10),0.5*np.ones(10),color='black',linestyle='--')
    fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure4_discrimination_generalization_%.3f_neu_%i_1_angle_ct.pdf'%(std_xor1,j),dpi=500,bbox_inches='tight')

    # XOR task
    fig=plt.figure(figsize=(1.5,1.8))
    ax=fig.add_subplot(1,1,1)
    functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.set_ylim([0.4,1.0])
    ax.set_ylabel('Decoding Performance')
    for i in range(len(model_vec)):
        ax.bar(-1.5*width+width*i,perf_xor_m[i,j],yerr=perf_xor_sem[i,j],width=width,color='green',alpha=alpha_vec[i])
    ax.plot(np.linspace(-0.5,0.5,10),0.5*np.ones(10),color='black',linestyle='--')
    fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure4_discrimination_generalization_%.3f_neu_%i_2_angle_ct.pdf'%(std_xor1,j),dpi=500,bbox_inches='tight')

    # Abstraction
    fig=plt.figure(figsize=(1.5,1.8))
    ax=fig.add_subplot(1,1,1)
    functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.set_ylim([0.4,1.0])
    ax.set_ylabel('Decoding Performance')
    for i in range(len(model_vec)):
        ax.bar(-1.5*width+width*i,perf_abs_m[i,j],yerr=perf_abs_sem[i,j],width=width,color='green',alpha=alpha_vec[i])
        ax.scatter(-1.5*width+width*i,perf_abs_c1_m[i,j],s=10,color='blue',edgecolor='black',linestyle='None')
        ax.scatter(-1.5*width+width*i,perf_abs_c2_m[i,j],s=10,color='green',edgecolor='black',linestyle='None')
        ax.scatter(-1.5*width+width*i,perf_abs_c3_m[i,j],s=10,color='red',edgecolor='black',linestyle='None')
    ax.plot(np.linspace(-0.5,0.5,10),0.5*np.ones(10),color='black',linestyle='--')
    fig.savefig('/home/ramon/Dropbox/chris_randy/plots/figure4_discrimination_generalization_%.3f_neu_%i_3_angle_ct.pdf'%(std_xor1,j),dpi=500,bbox_inches='tight')


         
# ss1=np.sum(feat_encoding_nonorm[:,v_used[h,0]]*dic_time['resol'],axis=1)[index_ct]
# ss2=np.sum(feat_encoding_nonorm[:,v_used[h,1]]*dic_time['resol'],axis=1)[index_ct]
# #index1=np.where(lin==1)[0]
# #index0=np.where(lin==0)[0]
# index1=np.where(xor==1)[0]
# index0=np.where(xor==0)[0]
# #index1=np.where(feat_binary_orig[:,1]==1)[0]
# #index0=np.where(feat_binary_orig[:,1]==0)[0]

# fig=plt.figure(figsize=(1,1))
# ax=fig.add_subplot(1,1,1)
# functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.set_xlabel('C1 Contacts')
# ax.set_ylabel('C3 Contacts')
# ax.set_xticks([0,4,8])
# ax.set_yticks([0,4,8])
# ax.scatter(ss1[index1],ss2[index1],color='orange',s=6,alpha=1)
# ax.scatter(ss1[index0],ss2[index0],color='green',s=6,alpha=1)
# fig.savefig('/home/ramon/Dropbox/chris_randy/plots/scatter_contacts_linear_xor_%s.pdf'%files_vec[i],dpi=500,bbox_inches='tight')
