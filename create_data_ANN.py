import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy.stats import sem
import scipy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
nan=float('nan')
import pickle as pkl
import nn_pytorch_decoders_recurrent_ANN
import miscellaneous_ANN
import functions_miscellaneous
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


###################################################################################################
# Parameters       
n_trials_train=400
n_trials_test=40
t_steps=30
t_dec=20
input_time=t_steps
xx=np.arange(t_steps)/10-2.0

batch_size=20
stim_nat='Poisson'
n_hidden=60
t_rnd_delay=10

sigma_train=1.0
sigma_test=1.0

#reg=2*1e-2
reg=1e-10
lr=0.005 #0.005

prob_vec={}
prob_vec['linear']=[[0.35,0.65],[0.35,0.65],[0.35,0.65]]
prob_vec['2d_xor']=[[0.3,0.7],[0.3,0.7],[0.3,0.7]]
prob_vec['xor']=[[0.23,0.77],[0.23,0.77],[0.23,0.77]]
#
# prob_vec['linear']=[[0.35,0.65],[0.35,0.65],[0.35,0.65]]
# prob_vec['2d_xor']=[[0.35,0.65],[0.35,0.65],[0.35,0.65]]
# prob_vec['xor']=[[0.35,0.65],[0.35,0.65],[0.35,0.65]]
# prob_vec['linear']=[[0.3,0.7],[0.3,0.7],[0.3,0.7]]
# prob_vec['2d_xor']=[[0.3,0.7],[0.3,0.7],[0.3,0.7]]
# prob_vec['xor']=[[0.3,0.7],[0.3,0.7],[0.3,0.7]]
#prob_vec['linear']=[[0.23,0.77],[0.23,0.77],[0.23,0.77]]
# prob_vec['2d_xor']=[[0.23,0.77],[0.23,0.77],[0.23,0.77]]
# prob_vec['xor']=[[0.23,0.77],[0.23,0.77],[0.23,0.77]]
prob_vec_train={}
prob_vec_train['linear']=[[0,1],[0,1],[0,1]]
prob_vec_train['2d_xor']=[[0,1],[0,1],[0,1]]
prob_vec_train['xor']=[[0,1],[0,1],[0,1]]

dim_in=len(prob_vec_train['linear'])
dim_stim=len(prob_vec_train['linear'][0])

task_vec=['linear'] #,'2d_xor', 'xor' 

n_files=10
n_rand=100
for i in range(len(task_vec)):
    print ('Task %s'%task_vec[i])
    net_perfs=nan*np.zeros((n_files,7,t_steps))
    abs_perfs_pre=nan*np.zeros((n_files,3,2,2,t_steps))
    abs_perfs_sh_pre=nan*np.zeros((n_files,n_rand,3,2,2))
    perf_task=nan*np.zeros((n_files,2,t_steps))
    for hh in range(n_files):
        print (hh)
        all_train=miscellaneous_ANN.create_input(stim_nat,n_trials_train,t_steps,input_time,prob_vec_train[task_vec[i]],w_rotation=0,t_rnd_delay=t_rnd_delay)
        all_test=miscellaneous_ANN.create_input(stim_nat,n_trials_test,t_steps,input_time,prob_vec[task_vec[i]],w_rotation=all_train['w_rotation'],t_rnd_delay=t_rnd_delay)
        if task_vec[i]=='linear':
            target_train=all_train['target_gen'][0]
            target_test=all_test['target_gen'][0]
        if task_vec[i]=='2d_xor':      
            target_train=all_train['target_2d_xor'][0]
            target_test=all_test['target_2d_xor'][0]
        if task_vec[i]=='xor':
            target_train=all_train['target_xor']
            target_test=all_test['target_xor']
        rec=nn_pytorch_decoders_recurrent_ANN.nn_recurrent(reg=reg,lr=lr,input_size=dim_in,output_size=2,n_layers=1,hidden_dim=n_hidden,t_dec=t_dec)
        rec.fit(input_seq=all_train['input_rec'],target_seq=target_train,batch_size=batch_size,sigma_noise=sigma_train)
    
        index0=np.where(target_test==0)[0]
        index1=np.where(target_test==1)[0]
        ut_train=rec.model(all_train['input_rec'],sigma_noise=sigma_train)[2].detach().numpy()
        ut_test=rec.model(all_test['input_rec'],sigma_noise=sigma_test)[2].detach().numpy()
        zt_train=rec.model(all_train['input_rec'],sigma_noise=sigma_train)[3].detach().numpy()
        zt_test=rec.model(all_test['input_rec'],sigma_noise=sigma_test)[3].detach().numpy()

        # Plot performance
        dec_train=np.argmax(zt_train,axis=2)
        dec_test=np.argmax(zt_test,axis=2) 
        reward_train=(np.ones(len(dec_train[:,t_dec]))-(target_train.detach().numpy()-dec_train[:,t_dec]))
        reward_test=(np.ones(len(dec_test[:,t_dec]))-(target_test.detach().numpy()-dec_test[:,t_dec]))
        
        # Classification Different Tasks
        supp0=LogisticRegression(C=1)#,class_weight='balanced',solver='lbfgs')
        mod0=supp0.fit(ut_train[:,t_dec],all_train['target_gen'][0])
        supp1=LogisticRegression(C=1)
        mod1=supp1.fit(ut_train[:,t_dec],all_train['target_gen'][1])
        supp2=LogisticRegression(C=1)
        mod2=supp2.fit(ut_train[:,t_dec],all_train['target_gen'][2])
        supp3=LogisticRegression(C=1)
        mod3=supp3.fit(ut_train[:,t_dec],all_train['target_2d_xor'][0])
        supp4=LogisticRegression(C=1)
        mod4=supp4.fit(ut_train[:,t_dec],all_train['target_2d_xor'][1])
        supp5=LogisticRegression(C=1)
        mod5=supp5.fit(ut_train[:,t_dec],all_train['target_2d_xor'][2])
        supp6=LogisticRegression(C=1)
        mod6=supp6.fit(ut_train[:,t_dec],all_train['target_xor'])
        
        # Train Abstraction Different Variables. 
        dic_class={}
        names_var=['c1','c2','c3']
        abs_var=[[1,2],[0,2],[0,1]]
        for f in range(len(abs_var)): # Loop over each variables (equivalent to loop over the two other variables)
            for ff in range(2): # Loop over the two variables in which we will train 
                index_cv=[all_train['target_gen'][abs_var[f][ff]]==1,all_train['target_gen'][abs_var[f][ff]]==0]
                for fff in range(2): # Loop over values of 1 and 0 for the trained variable
                    supp=LogisticRegression(C=1)
                    supp.fit(ut_train[index_cv[fff]][:,t_dec],all_train['target_gen'][f][index_cv[fff]])
                    dic_class['class_%s_tr_%i_%i'%(names_var[f],abs_var[f][ff],fff)]=supp
    
            
        for j in range(t_steps):
            # Different Tasks
            perf_task[hh,0,j]=(1-np.mean(abs(all_train['target_%s'%task_vec[i]].detach().numpy()-dec_train[:,j])))
            perf_task[hh,1,j]=(1-np.mean(abs(all_test['target_%s'%task_vec[i]].detach().numpy()-dec_test[:,j])))
            net_perfs[hh,0,j]=supp0.score(ut_test[:,j],all_test['target_gen'][0])
            net_perfs[hh,1,j]=supp1.score(ut_test[:,j],all_test['target_gen'][1])
            net_perfs[hh,2,j]=supp2.score(ut_test[:,j],all_test['target_gen'][2])
            net_perfs[hh,3,j]=supp3.score(ut_test[:,j],all_test['target_2d_xor'][0])
            net_perfs[hh,4,j]=supp4.score(ut_test[:,j],all_test['target_2d_xor'][1])
            net_perfs[hh,5,j]=supp5.score(ut_test[:,j],all_test['target_2d_xor'][2])
            net_perfs[hh,6,j]=supp6.score(ut_test[:,j],all_test['target_xor'])   
            
            #Test Different Abstractions. Should we use train or test variables?
            for f in range(len(abs_var)):
                for ff in range(2): 
                    index_cv=[all_test['target_gen'][abs_var[f][ff]]==0,all_test['target_gen'][abs_var[f][ff]]==1] # Note this is reversed wrt to train
                    for fff in range(2):
                        supp=dic_class['class_%s_tr_%i_%i'%(names_var[f],abs_var[f][ff],fff)]
                        abs_perfs_pre[hh,f,ff,fff,j]=supp.score(ut_test[index_cv[fff]][:,j],all_test['target_gen'][f][index_cv[fff]])
                        
        # Loop over null-hypothesis for CCGP. Everything evaluated only on T=20.
        # Calculate the permutation indices
        index_perm_all=nan*np.zeros((n_rand,3,2,2,2,n_hidden))
        for p in range(3):
            for pp in range(2):  
                for ppp in range(2): 
                    for pppp in range(2): 
                        for ppppp in range(n_rand):
                            index_perm_all[ppppp,p,pp,ppp,pppp]=np.random.permutation(np.arange(n_hidden))

        for nn in range(n_rand):
            #print (nn)
            for f in range(len(abs_var)): # Loop over each variables (equivalent to loop over the two other variables)
                for ff in range(2): # Loop over the two variables in which we will train 
                    var_cond_tr=(all_train['target_gen'][abs_var[f][ff]])
                    var_cond_te=(all_test['target_gen'][abs_var[f][ff]])
                    var_deco_tr=(all_train['target_gen'][f])
                    var_deco_te=(all_test['target_gen'][f])
                    index_cond=[var_cond_tr==1,var_cond_tr==0]
                    index_decod=[var_cond_te==0,var_cond_te==1]
                    
                    repr_sh_tr=nan*np.zeros((8*n_trials_train,n_hidden))
                    repr_sh_te=nan*np.zeros((8*n_trials_test,n_hidden))
                    for tt in range(2):
                        for ttt in range(2):
                            index_rot_tr=np.where((var_cond_tr==tt)&(var_deco_tr==ttt))
                            index_rot_te=np.where((var_cond_te==tt)&(var_deco_te==ttt))
                            index_perm=np.array(index_perm_all[nn,f,ff,tt,ttt],dtype=np.int16)
                            #repr_sh_tr[index_rot_tr]=ut_train[index_rot_tr][:,t_dec][:,index_perm]
                            #repr_sh_te[index_rot_te]=ut_test[index_rot_te][:,t_dec][:,index_perm]
                            repr_sh_tr[index_rot_tr]=ut_train[index_rot_tr][:,3][:,index_perm]
                            repr_sh_te[index_rot_te]=ut_test[index_rot_te][:,3][:,index_perm]
                            
                    for fff in range(2): # Loop over values of 1 and 0 for the trained variable
                        supp=LogisticRegression(C=1)
                        supp.fit(repr_sh_tr[index_cond[fff]],var_deco_tr[index_cond[fff]])
                        abs_perfs_sh_pre[hh,nn,f,ff,fff]=supp.score(repr_sh_te[index_decod[fff]],var_deco_te[index_decod[fff]])                
        
                
        ################################################
        dic={}
        dic['input_rec_train']=all_train['input_rec']
        dic['input_rec_test']=all_test['input_rec']
        dic['w_rotation']=all_train['w_rotation']
        
        dic['prob_vec']=prob_vec[task_vec[i]]
        dic['reg']=reg
        dic['lr']=lr

        dic['z_train']=zt_train
        dic['z_test']=zt_test
        dic['firing_rate_train']=ut_train
        dic['firing_rate_test']=ut_test
        dic['class_train']=target_train
        dic['class_test']=target_test
        
        # pathsave='/home/ramon/Dropbox/chris_randy/data/data_ANN/data_recurrent_task_%s_%i_%i_steps_%s_n_hidden_%i_dim_in_%i_t_delay_%i_noise_%.1f_n_test_%i_file_%i.pkl'%(task_vec[i],t_steps,t_dec,stim_nat,n_hidden,dim_in,t_rnd_delay,sigma_test,n_trials_test,hh)
        # datos=open(pathsave,'wb')
        # pkl.dump(dic,datos)
        # datos.close()

    perf_task_m=np.mean(perf_task,axis=0)           
    perf_task_sem=sem(perf_task,axis=0)        
    net_perfs_m=np.mean(net_perfs,axis=0)
    net_perfs_sem=sem(net_perfs,axis=0)
    
    abs_perfs=np.mean(abs_perfs_pre,axis=(2,3))
    abs_perfs_sh=np.mean(abs_perfs_sh_pre,axis=(3,4))
    abs_perfs_m=np.mean(abs_perfs,axis=(0))
    abs_perfs_sh_m=np.mean(abs_perfs_sh,axis=(0))
    abs_perfs_sem=sem(abs_perfs,axis=0)
    
    # Performance
    # fig=plt.figure(figsize=(2,2))
    # ax=fig.add_subplot(1,1,1)
    # functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    # colors=['green','green','green','blue','blue','blue','black']
    # alpha_vec=[0.3,0.6,1,0.3,0.6,1,1]
    # for ff in range(7): 
    #     ax.plot(xx,net_perfs_m[ff],color=colors[ff],alpha=alpha_vec[ff])#,linewidth=3)
    #     #ax.fill_between(xx,net_perfs_m[ff]-net_perfs_sem[ff],net_perfs_m[ff]+net_perfs_sem[ff],color=colors[ff],alpha=alpha_vec[ff])  
    # ax.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    # ax.axvline(0,color='black',linestyle='--')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Prob. Correct')
    # #plt.legend(loc='best')
    # ax.set_ylim([0.4,1.0])
    # fig.savefig('/home/ramon/Dropbox/chris_randy/plots/performance_easy_diff_tasks_ANNs_%s_noise_%.1f_%.1f_n_test_%i_suppl_noise_high.pdf'%(task_vec[i],sigma_train,sigma_test,n_trials_test),bbox_inches='tight',dpi=500)
    
    # # Abstraction
    # fig=plt.figure(figsize=(2,2))
    # ax=fig.add_subplot(1,1,1)
    # functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    # alpha_vec=[0.3,0.6,1]
    # for ff in range(3): 
    #     ax.plot(xx,abs_perfs_m[ff],color='brown',alpha=alpha_vec[ff])#,linewidth=3)
    #     #ax.fill_between(xx,net_perfs_m[ff]-net_perfs_sem[ff],net_perfs_m[ff]+net_perfs_sem[ff],color=colors[ff],alpha=alpha_vec[ff])  
    # ax.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    # ax.axvline(0,color='black',linestyle='--')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Prob. Correct')
    # #plt.legend(loc='best')
    # ax.set_ylim([0.4,1.0])
    # fig.savefig('/home/ramon/Dropbox/chris_randy/plots/abstraction_easy_diff_tasks_ANNs_%s_noise_%.1f_%.1f_n_test_%i_suppl_noise_high.pdf'%(task_vec[i],sigma_train,sigma_test,n_trials_test),bbox_inches='tight',dpi=500)
    
    
