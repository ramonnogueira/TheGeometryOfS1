import numpy as np
from scipy.stats import sem
import scipy.io
import os

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import svm
#from pyglmnet import GLM

from numpy.random import permutation
from numpy.random import choice
import imbalanced_data
import functions_miscellaneous
import nn_pytorch_encoders
import nn_pytorch_encoders_recurrent
from sklearn.preprocessing import StandardScaler

nan=float('nan')

# General Class for encoding models
class validation_encoding:
    def __init__(self,feat,target,lr_vec,reg,loss_type,n_val,n_cv):
        self.feat=feat
        self.lenf=len(self.feat[0,0])
        self.target=target
        self.lent=len(self.target[0,0])
        self.lr_vec=lr_vec
        self.reg=reg
        self.loss_type=loss_type
        self.n_splits_val=n_val
        self.n_splits=n_cv
        self.batch_size=64

class feedforward_validation(validation_encoding):
    def __init__(self,feat,target,lr_vec,reg,loss_type,n_val,n_cv,reward,corr):
        super().__init__(feat,target,lr_vec,reg,loss_type,n_val,n_cv,reward,corr)

    def validation(self,model,features_slow,dic_time):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True)
        perf=nan*np.zeros((3,2,self.lent,self.n_splits_val,len(self.lr_vec),self.n_splits)) # 0 train, 1 test, 2 validation        
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            print ('Validation ',g_val)
            val=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[val_index],features_general=self.feat[val_index],features_slow=features_slow,dic_time=dic_time,reward=reward[val_index])
            for hhh in range(len(self.lr_vec)):
                print ('  lr ',self.lr_vec[hhh])
                cv=KFold(n_splits=self.n_splits,shuffle=True) 
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    train=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[train_index1][train_index],features_general=self.feat[train_index1][train_index],features_slow=features_slow,dic_time=dic_time,reward=reward[train_index1][train_index])
                    test=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[train_index1][test_index],features_general=self.feat[train_index1][test_index],features_slow=features_slow,dic_time=dic_time,reward=reward[train_index1][test_index])
                    supp=nn_pytorch_encoders.nn_encoding(type_class=model,reg=self.reg,lr=self.lr_vec[hhh],loss_type=self.loss_type,len_feat=len(train['features'][0]),len_target=self.lent)
                    if self.corr=='none':
                        mod=supp.fit(train['features'],train['firing_rate'],batch_size=self.batch_size)
                        perf[0,:,:,g_val,hhh,g]=supp.score(train['features'],train['firing_rate'])
                        perf[1,:,:,g_val,hhh,g]=supp.score(test['features'],test['firing_rate'])
                        perf[2,:,:,g_val,hhh,g]=supp.score(val['features'],val['firing_rate'])
                    else:
                        mod=supp.fit_corr_incorr(train['features'],train['firing_rate'],batch_size=self.batch_size,reward=train['reward'],corr=corr)
                        perf[0,:,:,g_val,hhh,g]=supp.score_corr_incorr(train['features'],train['firing_rate'],train['reward'],corr=corr)
                        perf[1,:,:,g_val,hhh,g]=supp.score_corr_incorr(test['features'],test['firing_rate'],test['reward'],corr=corr)
                        perf[2,:,:,g_val,hhh,g]=supp.score_corr_incorr(val['features'],val['firing_rate'],val['reward'],corr=corr)
                    print (np.mean(perf[:,:,:,g_val,hhh,g],axis=2))
                    dic_models['g_val_%i_lr_%i_cv_%i'%(g_val,hhh,g)]=mod
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=5),'models':dic_models}
        return output
    
# Recurrent Class
class recurrent_validation(validation_encoding):
    def __init__(self,feat,target,lr_vec,reg,loss_type,n_val,n_cv,hidden_dim,nonlinearity):
        super().__init__(feat,target,lr_vec,reg,loss_type,n_val,n_cv)
        self.hidden_dim=hidden_dim
        self.nonlinearity=nonlinearity
        
    def validation(self):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True)
        perf=nan*np.zeros((3,2,self.lent,self.n_splits_val,len(self.lr_vec),self.n_splits)) # 0 train, 1 test, 2 validation        
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            print ('Validation ',g_val)
            for hhh in range(len(self.lr_vec)):
                print ('  lr ',self.lr_vec[hhh])
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    supp=nn_pytorch_encoders_recurrent.nn_encoding(loss_type=self.loss_type,reg=self.reg,lr=self.lr_vec[hhh],input_size=self.lenf,output_size=self.lent,hidden_dim=self.hidden_dim,n_layers=1,nonlinearity=self.nonlinearity)
                    model=supp.fit(self.feat[train_index1][train_index],self.target[train_index1][train_index],batch_size=self.batch_size)
                    perf[0,:,:,g_val,hhh,g]=supp.score(self.feat[train_index1][train_index],self.target[train_index1][train_index])
                    perf[1,:,:,g_val,hhh,g]=supp.score(self.feat[train_index1][test_index],self.target[train_index1][test_index])
                    perf[2,:,:,g_val,hhh,g]=supp.score(self.feat[val_index],self.target[val_index])
                    print (np.mean(perf[:,:,:,g_val,hhh,g],axis=2))
                    dic_models['g_val_%i_lr_%i_cv_%i'%(g_val,hhh,g)]=model
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=5),'models':dic_models}
        return output
    
class feedforward_validation_ANN(validation_encoding):
    def __init__(self,feat,target,lr_vec,reg,loss_type,n_val,n_cv):
        super().__init__(feat,target,lr_vec,reg,loss_type,n_val,n_cv)

    def validation(self,model,size_kernel):
        dic_models={}
        index_dic={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True)
        perf=nan*np.zeros((3,2,self.lent,self.n_splits_val,len(self.lr_vec),self.n_splits)) # 0 train, 1 test, 2 validation
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            print ('Validation ',g_val)
            index_dic['train1_%i'%g_val]=train_index1
            index_dic['val_%i'%g_val]=val_index
            val=functions_miscellaneous.crear_encoding_feedforward_ANN(firing_rate=self.target[val_index],features=self.feat[val_index],size_kernel=size_kernel)
            for hhh in range(len(self.lr_vec)):
                print ('  lr ',self.lr_vec[hhh])
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    index_dic['train_%i_%i'%(g_val,g)]=train_index
                    index_dic['test_%i_%i'%(g_val,g)]=test_index
                    train=functions_miscellaneous.crear_encoding_feedforward_ANN(firing_rate=self.target[train_index1][train_index],features=self.feat[train_index1][train_index],size_kernel=size_kernel)
                    test=functions_miscellaneous.crear_encoding_feedforward_ANN(firing_rate=self.target[train_index1][test_index],features=self.feat[train_index1][test_index],size_kernel=size_kernel)
                    supp=nn_pytorch_encoders.nn_encoding(type_class=model,reg=self.reg,lr=self.lr_vec[hhh],loss_type=self.loss_type,len_feat=len(train['features'][0]),len_target=self.lent)
                    mod=supp.fit(train['features'],train['firing_rate'],batch_size=self.batch_size)
                    perf[0,:,:,g_val,hhh,g]=supp.score(train['features'],train['firing_rate'])
                    perf[1,:,:,g_val,hhh,g]=supp.score(test['features'],test['firing_rate'])
                    perf[2,:,:,g_val,hhh,g]=supp.score(val['features'],val['firing_rate'])
                    print ('    ',np.mean(perf[:,:,:,g_val,hhh,g],axis=2))
                    dic_models['g_val_%i_lr_%i_cv_%i'%(g_val,hhh,g)]=mod
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=5),'models':dic_models,'index':index_dic}
        return output

###################################################################################
class linear_model_validation:
   def __init__(self,feat,target,n_val,n_cv):
       self.feat=feat
       self.lenf=len(self.feat[0])
       self.target=target
       self.lent=len(self.target[0,0])
       self.n_splits_val=n_val
       self.n_splits=n_cv
       
   def linregress(self,features_slow,dic_time):
       cv_val=KFold(n_splits=self.n_splits_val,shuffle=True)
       perf_pre=nan*np.zeros((3,self.lent,self.n_splits_val,self.n_splits)) # 0 para train, 1 para test, 2 para validation
       wei=nan*np.zeros((self.lent,self.n_splits_val,self.n_splits,298)) # Cuidado
       g_val=0
       for train_index1, val_index in cv_val.split(self.feat):
           cv=KFold(n_splits=self.n_splits,shuffle=True)
           val=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[val_index],features_general=self.feat[val_index],features_slow=features_slow,dic_time=dic_time)
           g=0
           for train_index, test_index in cv.split(self.feat[train_index1]):
               print (g)
               train=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[train_index1][train_index],features_general=self.feat[train_index1][train_index],features_slow=features_slow,dic_time=dic_time)
               test=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[train_index1][test_index],features_general=self.feat[train_index1][test_index],features_slow=features_slow,dic_time=dic_time)
               #lr=LinearRegression()
               lr=Ridge(alpha=500)
               fit=lr.fit(train['features'],train['firing_rate'])
               perf_pre[0,:,g_val,g]=fit.score(train['features'],train['firing_rate'])
               perf_pre[1,:,g_val,g]=fit.score(test['features'],test['firing_rate'])
               perf_pre[2,:,g_val,g]=fit.score(val['features'],val['firing_rate'])
               print (np.mean(perf_pre[:,:,g_val,g],axis=1))
               wei[:,g_val,g]=lr.coef_
               g=(g+1)                
           g_val=(g_val+1)
       performance=np.nanmean(perf_pre,axis=(2,3))
       weights=np.nanmean(wei,axis=(1,2))
       output={'performance':performance,'weights':weights}
       return output

class glm_validation:
    def __init__(self,feat,target,n_val,n_cv):
        self.feat=feat
        self.lenf=len(self.feat[0])
        self.target=target
        self.lent=len(self.target[0,0])
        self.n_splits_val=n_val
        self.n_splits=n_cv
        
    def glm(self,features_slow,dic_time):
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True)
        perf_pre=nan*np.zeros((3,self.n_splits_val,self.n_splits)) # 0 para train, 1 para test, 2 para validation
        wei=nan*np.zeros((self.n_splits_val,self.n_splits,298)) # Cuidado
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
           cv=KFold(n_splits=self.n_splits,shuffle=True)
           val=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[val_index],features_general=self.feat[val_index],features_slow=features_slow,dic_time=dic_time)
           g=0
           for train_index, test_index in cv.split(self.feat[train_index1]):
               train=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[train_index1][train_index],features_general=self.feat[train_index1][train_index],features_slow=features_slow,dic_time=dic_time)
               test=functions_miscellaneous.crear_encoding_feedforward(firing_rate_general=self.target[train_index1][test_index],features_general=self.feat[train_index1][test_index],features_slow=features_slow,dic_time=dic_time)
               glm=GLM(distr='poisson',score_metric='pseudo_R2',alpha=0,reg_lambda=1)
               glm.fit(train['features'],train['firing_rate'][:,0])
               perf_pre[0,g_val,g]=glm.score(train['features'],train['firing_rate'][:,0])
               perf_pre[1,g_val,g]=glm.score(test['features'],test['firing_rate'][:,0])
               perf_pre[2,g_val,g]=glm.score(val['features'],val['firing_rate'][:,0])
               g=g+1
           g_val=(g_val+1)
        performance=np.nanmean(perf_pre,axis=(1,2))
        weights=np.nanmean(wei,axis=(0,1))
        output={'performance':performance,'weights':weights}
        return output


