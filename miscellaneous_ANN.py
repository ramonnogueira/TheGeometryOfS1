import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import itertools
nan=float('nan')


def create_io_rnd(io_gen,n):
    io_rnd=nan*np.zeros((n,len(io_gen[0])))
    for i in range(n):
        io_rnd[i]=np.random.permutation(io_gen[0])
    return io_rnd

def create_gen_xor(io_gen):
    all_binary=np.array(np.transpose(io_gen),dtype=np.int16)
    all_col=np.sum(all_binary,axis=1)
    io_xor=all_col%2
    return io_xor

def create_2d_xor(io_gen):
    pairs=[[0,1],[0,2],[1,2]]
    io_xor=nan*np.zeros((len(pairs),len(io_gen[0])))
    for i in range(len(pairs)):
        all_binary=np.array(np.transpose(io_gen)[:,pairs[i]],dtype=np.int16)
        all_col=np.sum(all_binary,axis=1)
        io_xor[i]=all_col%2
    return io_xor

#################################################################################
    
def create_input(stim_nat,n_trials,t_steps,input_time,prob_vec,w_rotation,t_rnd_delay): # n_dim =1 is not well implemented, be careful #io_nlin,
    # All possible combinations of inputs
    n_dim=len(prob_vec)
    if n_dim==1:
        comb_inp=[np.array(seq) for seq in itertools.product(prob_vec[0])]
        io_gen_pre=[np.array(seq) for seq in itertools.product([0,1])]
    if n_dim==2:
        comb_inp=[np.array(seq) for seq in itertools.product(prob_vec[0],prob_vec[1])]
        io_gen_pre=[np.array(seq) for seq in itertools.product([0,1],[0,1])]
    if n_dim==3:
        comb_inp=[np.array(seq) for seq in itertools.product(prob_vec[0],prob_vec[1],prob_vec[2])]
        io_gen_pre=[np.array(seq) for seq in itertools.product([0,1],[0,1],[0,1])]
    if n_dim==4:
        comb_inp=[np.array(seq) for seq in itertools.product(prob_vec[0],prob_vec[1],prob_vec[2],prob_vec[3])]
        io_gen_pre=[np.array(seq) for seq in itertools.product([0,1],[0,1],[0,1],[0,1])]
    if n_dim==5:
        comb_inp=[np.array(seq) for seq in itertools.product(prob_vec[0],prob_vec[1],prob_vec[2],prob_vec[3],prob_vec[4])]
        io_gen_pre=[np.array(seq) for seq in itertools.product([0,1],[0,1],[0,1],[0,1],[0,1])]
    comb_inp=np.array(comb_inp)
    
    # Output functions. Linear, Non-linear and XOR
    io_gen=np.transpose(np.array(io_gen_pre,dtype=bool))
    io_lin=io_gen[0]
    n_rnd=100
    io_rnd=create_io_rnd(io_gen=io_gen,n=n_rnd)
    io_xor=create_gen_xor(io_gen)
    io_2dx=create_2d_xor(io_gen)
    
    # The actual stimuli sequences
    dic={}
    input_vec_orig=np.zeros(((2**n_dim)*n_trials,t_steps,n_dim))
    target_gen_vec_pre=nan*np.zeros((len(io_gen),(2**n_dim)*n_trials))
    target_rnd_vec_pre=nan*np.zeros((n_rnd,(2**n_dim)*n_trials))
    target_lin_vec_pre=nan*np.zeros(((2**n_dim)*n_trials))
    target_xor_vec_pre=nan*np.zeros(((2**n_dim)*n_trials))
    target_2dx_vec_pre=nan*np.zeros((3,(2**n_dim)*n_trials))
    
    # Create Stimulus on the original space. We will use Gaussian or Binomial values
    for k in range(2**n_dim):
        for i in range(n_trials):
            for ii in range(input_time):
                 for iii in range(n_dim): 
                     if stim_nat=='Poisson':
                         input_vec_orig[(k*n_trials+i),ii,iii]=np.random.binomial(1,comb_inp[k][iii])
                     if stim_nat=='Gaussian':
                         input_vec_orig[(k*n_trials+i),ii,iii]=np.random.normal(loc=comb_inp[k][iii])
        target_lin_vec_pre[k*n_trials:(k+1)*n_trials]=io_lin[k]
        target_xor_vec_pre[k*n_trials:(k+1)*n_trials]=io_xor[k]
        for gg in range(len(io_gen)):
            target_gen_vec_pre[gg,k*n_trials:(k+1)*n_trials]=io_gen[gg,k]
        for gg in range(len(io_2dx)):
            target_2dx_vec_pre[gg,k*n_trials:(k+1)*n_trials]=io_2dx[gg,k]
        for gg in range(n_rnd):
            target_rnd_vec_pre[gg,k*n_trials:(k+1)*n_trials]=io_rnd[gg,k]
            
    # We rotate the original space. We use a orthonormal matrix
    if np.shape(w_rotation)==():
        w_rotation=scipy.stats.ortho_group.rvs(n_dim)
    input_vec_pre=nan*np.zeros(((2**n_dim)*n_trials,t_steps,n_dim))
    for i in range(t_steps):
        input_vec_pre[:,i]=np.dot(w_rotation,input_vec_orig[:,i].T).T
            
    # We create the random delay on each trial
    for i in range(len(input_vec_pre)):
        num_beg=int(np.random.choice(np.arange(t_rnd_delay),1))
        #num_beg=0
        input_vec_pre[i,0:(num_beg)]=0 #zeros from beginning
        input_vec_pre[i,(t_steps-t_rnd_delay+num_beg+1):]=0 #zeros from end
        
    # Shuffle indeces
    index_def=np.random.permutation(np.arange((2**n_dim)*n_trials))
    input_vec=input_vec_pre[index_def]
    target_gen_vec=target_gen_vec_pre[:,index_def]
    target_rnd_vec=target_rnd_vec_pre[:,index_def]
    target_2dx_vec=target_2dx_vec_pre[:,index_def]
    target_lin_vec=target_lin_vec_pre[index_def]
    target_xor_vec=target_xor_vec_pre[index_def]
    # Return input and target in torch format
    dic['input_rec']=Variable(torch.from_numpy(np.array(input_vec,dtype=np.float32)),requires_grad=False)
    dic['target_gen']=Variable(torch.from_numpy(np.array(target_gen_vec,dtype=np.int16)),requires_grad=False)
    dic['target_rnd']=Variable(torch.from_numpy(np.array(target_rnd_vec,dtype=np.int16)),requires_grad=False)
    dic['target_linear']=Variable(torch.from_numpy(np.array(target_lin_vec,dtype=np.int16)),requires_grad=False)
    dic['target_2d_xor']=Variable(torch.from_numpy(np.array(target_2dx_vec,dtype=np.int16)),requires_grad=False)
    dic['target_xor']=Variable(torch.from_numpy(np.array(target_xor_vec,dtype=np.int16)),requires_grad=False)
    dic['io_gen']=io_gen
    dic['io_linear']=io_lin
    dic['io_xor']=io_xor
    dic['io_2dx']=io_2dx
    dic['w_rotation']=w_rotation
    return dic

################################################################################################
def dim_cv(n_split,data):
    num_trials=len(data)
    num_neu=len(data[0])
    perfor=np.zeros((2,n_split,num_neu))
    perfor_cum=np.zeros((2,n_split,num_neu))
    dic={}
    
    #cv=ShuffleSplit(n_splits=n_split,test_size=0.2)
    cv=KFold(n_splits=n_split)
    g=0
    for train_index, test_index in cv.split(data):
        x_train=data[train_index]
        x_test=data[test_index]
        
        # Explained Variance Train
        pca_clf=PCA(n_components=num_neu)
        pca=pca_clf.fit(x_train)
       
        # Project test and train set onto eigenvectors from train set
        proj_data_train=np.transpose(np.dot(pca.components_,np.transpose(x_train)))
        cov_train=np.cov(proj_data_train,rowvar=False)
        proj_data_test=np.transpose(np.dot(pca.components_,np.transpose(x_test)))
        cov_test=np.cov(proj_data_test,rowvar=False)
        
        # Total Variance
        var_total_train=np.sum(cov_train.diagonal())
        var_total_test=np.sum(cov_test.diagonal())
        
        # Variance explained on the test and train set
        var_train=np.zeros(num_neu)
        var_test=np.zeros(num_neu)
        var_train_cum=np.zeros(num_neu)
        var_test_cum=np.zeros(num_neu)
        
        for jj in range(num_neu):
            var_train[jj]=(cov_train.diagonal()[jj])/var_total_train
            var_test[jj]=(cov_test.diagonal()[jj])/var_total_test     
            var_train_cum[jj]=(np.sum(cov_train.diagonal()[0:(jj+1)])/var_total_train)      
            var_test_cum[jj]=(np.sum(cov_test.diagonal()[0:(jj+1)])/var_total_test)      
        
        perfor[0,g]=var_train
        perfor[1,g]=var_test
        perfor_cum[0,g]=var_train_cum
        perfor_cum[1,g]=var_test_cum
        
        g=g+1 
    dic['perfor']=perfor
    dic['perfor_cum']=perfor_cum
    return dic

def structure_index(x):
    dim_pre=np.sum(x)/float(len(x))
    if dim_pre<0.5:
        dim_pre=0.5
    return 2*(1-dim_pre)

def rank(x,eps):
    '''
    Given a cummulative explained variance curve, it outputs the rank
    '''
    rest=(1.0-x)
    rank=len(rest[rest>eps])+1
    return rank