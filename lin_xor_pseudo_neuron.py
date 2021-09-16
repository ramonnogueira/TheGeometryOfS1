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
n_rand=100
n_sh=1
n_rot=100
perc_tr=0.8
n_tr=int(perc_tr*nt_used)
n_te=int(nt_used-n_tr)
v_used=np.array([[1,2],[1,3],[2,3]])
t_ini=0
num_steps=(num_steps_pre-t_ini)
reg=1e-10
#neu_vec=np.array(np.logspace(1,3.765,10),dtype=np.int16)
neu_vec=np.array([5840])
pertur_vec=np.logspace(-2,0.6,10)

task_train=nan*np.zeros((2*n_tr))
task_test=nan*np.zeros((2*n_te))
task_vec=np.array([1,0])
for jj in range(2): # Loop on index 0 and 1 for a given task
    task_train[jj*n_tr:(jj+1)*n_tr]=task_vec[jj]*np.ones(n_tr)
    task_test[jj*n_te:(jj+1)*n_te]=task_vec[jj]*np.ones(n_te)

perf_lin=nan*np.zeros((n_rand,len(v_used),len(neu_vec),2))
perf_xor=nan*np.zeros((n_rand,len(v_used),len(neu_vec),2))
perf_lin_sh=nan*np.zeros((n_rand,len(v_used),n_sh,2))
perf_xor_sh=nan*np.zeros((n_rand,len(v_used),n_sh,2))
perf_lin_pert=nan*np.zeros((n_rand,len(v_used),len(neu_vec),len(pertur_vec),2))
perf_xor_pert=nan*np.zeros((n_rand,len(v_used),len(neu_vec),len(pertur_vec),2))
perf_lin_rot=nan*np.zeros((n_rand,len(v_used),n_rot,2))
perf_xor_rot=nan*np.zeros((n_rand,len(v_used),n_rot,2))
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

        pop_lin_tr=nan*np.zeros((len(v_used),2*n_tr,num_steps*neu_f))
        pop_lin_te=nan*np.zeros((len(v_used),2*n_te,num_steps*neu_f))
        pop_xor_tr=nan*np.zeros((len(v_used),2*n_tr,num_steps*neu_f))
        pop_xor_te=nan*np.zeros((len(v_used),2*n_te,num_steps*neu_f))

        for h in range(len(v_used)):
            feat_sum=np.sum(feat_pre[:,v_used[h]],axis=(1,2)) # Cuidado!!
            index_ct=(feat_sum!=0)
            feat_used=feat_pre[index_ct]
            firing_used=firing_rate[index_ct]

            m_orto=ortho_group.rvs(dim=2)
            create_xor=create_xor_median(feat=feat_used,v_used=v_used[h],m_orto=m_orto)
            xor=create_xor['xor']
            lin=create_xor['lin']

            index_lin1=np.where(lin==1)[0]
            index_lin0=np.where(lin==0)[0]
            index_xor1=np.where(xor==1)[0]
            index_xor0=np.where(xor==0)[0]
            index_all=[[index_lin1,index_lin0],[index_xor1,index_xor0]]
   
            for jj in range(2):
                skf=ShuffleSplit(n_splits=1,train_size=perc_tr)
                for train, test in skf.split(index_all[0][jj]):
                    ind_lin_tr=np.random.choice(index_all[0][jj][train],n_tr,replace=True)
                    ind_lin_te=np.random.choice(index_all[0][jj][test],n_te,replace=True)
                    pop_lin_tr[h,jj*n_tr:(jj+1)*n_tr]=np.reshape(firing_used[ind_lin_tr],(n_tr,-1))
                    pop_lin_te[h,jj*n_te:(jj+1)*n_te]=np.reshape(firing_used[ind_lin_te],(n_te,-1))
                skf=ShuffleSplit(n_splits=1,train_size=perc_tr)
                for train, test in skf.split(index_all[1][jj]):
                    ind_xor_tr=np.random.choice(index_all[1][jj][train],n_tr,replace=True)
                    ind_xor_te=np.random.choice(index_all[1][jj][test],n_te,replace=True)
                    pop_xor_tr[h,jj*n_tr:(jj+1)*n_tr]=np.reshape(firing_used[ind_xor_tr],(n_tr,-1))
                    pop_xor_te[h,jj*n_te:(jj+1)*n_te]=np.reshape(firing_used[ind_xor_te],(n_te,-1))

        if (i==0):
            pseudo_lin_tr=pop_lin_tr.copy()
            pseudo_lin_te=pop_lin_te.copy()
            pseudo_xor_tr=pop_xor_tr.copy()
            pseudo_xor_te=pop_xor_te.copy()
        else:
            pseudo_lin_tr=np.concatenate((pseudo_lin_tr,pop_lin_tr),axis=2)
            pseudo_lin_te=np.concatenate((pseudo_lin_te,pop_lin_te),axis=2)
            pseudo_xor_tr=np.concatenate((pseudo_xor_tr,pop_xor_tr),axis=2)
            pseudo_xor_te=np.concatenate((pseudo_xor_te,pop_xor_te),axis=2)

    # Decoding
    for h in range(len(v_used)):
        print ('  ',h)
        for ll in range(len(neu_vec)):
            neu_rnd=np.random.choice(np.arange(len(pseudo_lin_tr[0,0])),neu_vec[ll],replace=False)
            #supp_lin=LinearSVC(C=reg,class_weight='balanced')
            supp_lin=LogisticRegression(C=reg,class_weight='balanced')
            mod_lin=supp_lin.fit(pseudo_lin_tr[h][:,neu_rnd],task_train)
            perf_lin[r,h,ll,0]=supp_lin.score(pseudo_lin_tr[h][:,neu_rnd],task_train)
            perf_lin[r,h,ll,1]=supp_lin.score(pseudo_lin_te[h][:,neu_rnd],task_test)
            #supp_xor=LinearSVC(C=reg,class_weight='balanced')
            supp_xor=LogisticRegression(C=reg,class_weight='balanced')
            mod_xor=supp_xor.fit(pseudo_xor_tr[h][:,neu_rnd],task_train)
            perf_xor[r,h,ll,0]=supp_xor.score(pseudo_xor_tr[h][:,neu_rnd],task_train)
            perf_xor[r,h,ll,1]=supp_xor.score(pseudo_xor_te[h][:,neu_rnd],task_test)

            for hh in range(len(pertur_vec)):
                pseudo_lin_tr_pert=nan*np.zeros(np.shape(pseudo_lin_tr[h]))
                pseudo_lin_te_pert=nan*np.zeros(np.shape(pseudo_lin_te[h]))
                pseudo_xor_tr_pert=nan*np.zeros(np.shape(pseudo_xor_tr[h]))
                pseudo_xor_te_pert=nan*np.zeros(np.shape(pseudo_xor_te[h]))
                for hhh in range(2):
                    index_hhh_tr=np.where(task_train==hhh)[0]
                    index_hhh_te=np.where(task_test==hhh)[0]
                    pert=np.random.normal(0,pertur_vec[hh],(len(pseudo_lin_tr[0,0])))
                    pseudo_lin_tr_pert[index_hhh_tr]=(pseudo_lin_tr[h,index_hhh_tr]+pert)
                    pseudo_lin_te_pert[index_hhh_te]=(pseudo_lin_te[h,index_hhh_te]+pert)
                    pseudo_xor_tr_pert[index_hhh_tr]=(pseudo_xor_tr[h,index_hhh_tr]+pert)
                    pseudo_xor_te_pert[index_hhh_te]=(pseudo_xor_te[h,index_hhh_te]+pert)
                
                supp_lin=LogisticRegression(C=reg,class_weight='balanced')
                mod_lin=supp_lin.fit(pseudo_lin_tr_pert[:,neu_rnd],task_train)
                perf_lin_null[r,h,ll,hh,0]=supp_lin.score(pseudo_lin_tr_pert[:,neu_rnd],task_train)
                perf_lin_null[r,h,ll,hh,1]=supp_lin.score(pseudo_lin_te_pert[:,neu_rnd],task_test)
                
                supp_xor=LogisticRegression(C=reg,class_weight='balanced')
                mod_xor=supp_xor.fit(pseudo_xor_tr_pert[:,neu_rnd],task_train)
                perf_xor_null[r,h,ll,hh,0]=supp_xor.score(pseudo_xor_tr_pert[:,neu_rnd],task_train)
                perf_xor_null[r,h,ll,hh,1]=supp_xor.score(pseudo_xor_te_pert[:,neu_rnd],task_test)

        # for ss in range(n_sh):
        #     clase_tr_sh=permutation(task_train)
        #     clase_te_sh=permutation(task_test)
        #     supp_lin=LinearSVC(C=reg,class_weight='balanced')
        #     #supp_lin=LogisticRegression(C=reg,class_weight='balanced')
        #     mod_lin=supp_lin.fit(pseudo_lin_tr[h],clase_tr_sh)
        #     perf_lin_sh[r,h,ss,0]=supp_lin.score(pseudo_lin_tr[h],clase_tr_sh)
        #     perf_lin_sh[r,h,ss,1]=supp_lin.score(pseudo_lin_te[h],clase_te_sh)
        #     supp_xor=LinearSVC(C=reg,class_weight='balanced')
        #     #supp_xor=LogisticRegression(C=reg,class_weight='balanced')
        #     mod_xor=supp_xor.fit(pseudo_xor_tr[h],task_train)
        #     perf_xor_sh[r,h,ss,0]=supp_xor.score(pseudo_xor_tr[h],clase_tr_sh)
        #     perf_xor_sh[r,h,ss,1]=supp_xor.score(pseudo_xor_te[h],clase_te_sh)
            

    #print (np.mean(perf_lin,axis=(0)))
    print (np.nanmean(perf_lin,axis=(0,1)))
    #print (np.nanstd(perf_lin,axis=(0,1)))
    #print (np.mean(perf_xor,axis=(0)))
    print (np.nanmean(perf_xor,axis=(0,1)))
    #print (np.nanstd(perf_xor,axis=(0,1)))
    print (np.nanmean(perf_lin_null,axis=(0,1,2)))
    print (np.nanmean(perf_xor_null,axis=(0,1,2)))

    
perf_lin_m=np.nanmean(perf_lin,axis=(0,1))
perf_lin_err=np.nanstd(perf_lin,axis=(0,1))
perf_xor_m=np.nanmean(perf_xor,axis=(0,1))
perf_xor_err=np.nanstd(perf_xor,axis=(0,1))

perf_lin_null_m=np.nanmean(perf_lin_null,axis=(0,1))
perf_lin_null_err=np.nanstd(perf_lin_null,axis=(0,1))
perf_xor_null_m=np.nanmean(perf_xor_null,axis=(0,1))
perf_xor_null_err=np.nanstd(perf_xor_null,axis=(0,1))

#############################################
# Plots
fig=plt.figure(figsize=(3,3))
ax=fig.add_subplot(1,1,1)
functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.set_ylim([0.4,1.0])
ax.set_ylabel('Decoding Performance')
ax.set_xlabel('Perturbation Strength ($\sigma$)')
ax.plot(pertur_vec,perf_lin_null_m[0,:,1],color='blue')
ax.fill_between(pertur_vec,perf_lin_null_m[0,:,1]-perf_lin_null_err[0,:,1],perf_lin_null_m[0,:,1]+perf_lin_null_err[0,:,1],color='blue',alpha=0.5)
ax.plot(pertur_vec,perf_xor_null_m[0,:,1],color='red')
ax.fill_between(pertur_vec,perf_xor_null_m[0,:,1]-perf_xor_null_err[0,:,1],perf_xor_null_m[0,:,1]+perf_xor_null_err[0,:,1],color='red',alpha=0.5)
ax.plot(pertur_vec,0.5*np.ones(len(pertur_vec)),color='black',linestyle='--')
ax.set_xscale('log')
fig.savefig('/home/ramon/Dropbox/chris_randy/plots/lin_xor_perturbation.pdf',dpi=500,bbox_inches='tight')

# plt.hist(perf_lin[:,0,1],color='blue',alpha=0.5)
# plt.axvline(np.mean(perf_lin[:,0,1]),color='blue',alpha=0.5)
# plt.hist(perf_lin[:,1,1],color='green',alpha=0.5)
# plt.axvline(np.mean(perf_lin[:,1,1]),color='green',alpha=0.5)
# plt.hist(perf_lin[:,2,1],color='red',alpha=0.5)
# plt.axvline(np.mean(perf_lin[:,2,1]),color='red',alpha=0.5)
# plt.show()

# plt.hist(perf_xor[:,0,1],color='blue',alpha=0.5)
# plt.axvline(np.mean(perf_xor[:,0,1]),color='blue',alpha=0.5)
# plt.hist(perf_xor[:,1,1],color='green',alpha=0.5)
# plt.axvline(np.mean(perf_xor[:,1,1]),color='green',alpha=0.5)
# plt.hist(perf_xor[:,2,1],color='red',alpha=0.5)
# plt.axvline(np.mean(perf_xor[:,2,1]),color='red',alpha=0.5)
# plt.show()


