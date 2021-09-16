import numpy as np
import matplotlib.pylab as plt
from scipy.stats import sem
import scipy.io
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from numpy.random import permutation
from numpy.random import choice
from sklearn.linear_model import LinearRegression
import imbalanced_data
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
import scipy.stats
#import ipdb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import nn_pytorch_decoders
import nn_pytorch_recurrent
from sklearn import preprocessing
nan=float('nan')


class logregress_validation:
    def __init__(self,feat,clase,reg,lr_vec,n_val,n_cv):
        self.feat=feat
        self.lenf=len(self.feat[0,0]) # Remember that the format needs to be for convolutional
        self.lent=len(self.feat[0,0,0])
        self.clase=clase
        self.reg=reg
        self.lr_vec=lr_vec
        self.n_splits_val=n_val
        self.n_splits=n_cv
        self.batch_size=64
        
    def logregress(self,model,reward,corr):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True) # 0 para train, 1 para test, 2 para validation                   
        perf=nan*np.zeros((3,self.n_splits_val,len(self.lr_vec),self.n_splits))
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            #print ('Validation ',g_val)
            for hhh in range(len(self.lr_vec)):
                #print ('  lr ',self.lr_vec[hhh])
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    # Feedforward or Recurrent
                    if model=='recurrent':
                        supp=nn_pytorch_recurrent.nn_recurrent(reg=self.reg,lr=self.lr_vec[hhh],input_size=self.lenf,hidden_dim=self.n_hidden_units,nonlinearity=self.nonlinearity)
                    else:
                        supp=nn_pytorch_decoders.nn_feedforward(type_class=model,reg=self.reg,lr=self.lr_vec[hhh],len_feat=self.lenf,len_time=self.lent)
                    # All trials or correct vs incorrect
                    if corr=='none':
                        mod=supp.fit(feat=self.feat[train_index1][train_index],clase=self.clase[train_index1][train_index],batch_size=self.batch_size,reward=reward[train_index1][train_index])
                        perf[0,g_val,hhh,g]=supp.score(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                        perf[1,g_val,hhh,g]=supp.score(self.feat[train_index1][test_index],self.clase[train_index1][test_index])
                        perf[2,g_val,hhh,g]=supp.score(self.feat[val_index],self.clase[val_index])
                        #print ('    ',perf[:,g_val,hhh,g])
                    else:
                        mod=supp.fit_corr_incorr(feat=self.feat[train_index1][train_index],clase=self.clase[train_index1][train_index],batch_size=self.batch_size,reward=reward[train_index1][train_index],corr=corr)
                        perf[0,g_val,hhh,g]=supp.score_corr_incorr(self.feat[train_index1][train_index],self.clase[train_index1][train_index],reward=reward[train_index1][train_index],corr=corr)
                        perf[1,g_val,hhh,g]=supp.score_corr_incorr(self.feat[train_index1][test_index],self.clase[train_index1][test_index],reward=reward[train_index1][test_index],corr=corr)
                        perf[2,g_val,hhh,g]=supp.score_corr_incorr(self.feat[val_index],self.clase[val_index],reward=reward[val_index],corr=corr)
                        #print ('    ',perf[:,g_val,hhh,g])
                    dic_models['g_val_%i_lr_%i_cv_%i'%(g_val,hhh,g)]=mod
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=3),'models':dic_models}
        return output

    def logregress_ANN(self,model):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True) # 0 para train, 1 para test, 2 para validation                   
        perf=nan*np.zeros((3,self.n_splits_val,len(self.lr_vec),self.n_splits))
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            print ('Validation ',g_val)
            for hhh in range(len(self.lr_vec)):
                print ('  lr ',self.lr_vec[hhh])
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    supp=nn_pytorch_decoders.nn_feedforward(type_class=model,reg=self.reg,lr=self.lr_vec[hhh],len_feat=self.lenf,len_time=self.lent)
                    mod=supp.fit_no_equal(feat=self.feat[train_index1][train_index],clase=self.clase[train_index1][train_index],batch_size=self.batch_size)
                    perf[0,g_val,hhh,g]=supp.score(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[1,g_val,hhh,g]=supp.score(self.feat[train_index1][test_index],self.clase[train_index1][test_index])
                    perf[2,g_val,hhh,g]=supp.score(self.feat[val_index],self.clase[val_index])
                    print ('    ',perf[:,g_val,hhh,g])
                    dic_models['g_val_%i_lr_%i_cv_%i'%(g_val,hhh,g)]=mod
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=3),'models':dic_models}
        return output

class feedforward_validation(logregress_validation):
    def __init__(self,feat,clase,reg,lr_vec,n_val,n_cv):
        super().__init__(feat,clase,reg,lr_vec,n_val,n_cv)

class recurrent_validation(logregress_validation):
    def __init__(self,feat,clase,reg,lr_vec,n_val,n_cv,n_hidden_units,nonlinearity):
        super().__init__(feat,clase,reg,lr_vec,n_val,n_cv)
        self.n_hidden_units=n_hidden_units
        self.nonlinearity=nonlinearity

    



