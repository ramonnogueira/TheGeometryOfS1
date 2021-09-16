import os
#import matplotlib.pylab as plt
import numpy as np
import scipy
#import ipdb
import math
import sys
#import tables
import pandas
import sys
from scipy.stats import sem
from scipy.stats import pearsonr
nan=float('nan')

class spikes():
    def __init__(self,rwin_time,spikes_raw,neu_ident,dic_time): 
        self.rwin_time=rwin_time
        self.spikes_raw=spikes_raw
        self.neu_ident=neu_ident
        self.neu_uniq=np.unique(self.neu_ident)
        self.num_trials=len(self.rwin_time)
        #
        self.start_window=dic_time['start_window']
        self.end_window=dic_time['end_window']
        self.resol=dic_time['resol']
        self.num_steps=int(np.round((self.end_window-self.start_window)/self.resol))
        
    def rate_time(self):
        rate=nan*np.zeros((self.num_trials,len(self.neu_uniq),self.num_steps))
        for i in range(len(self.neu_uniq)):
            ind_spk_neu=np.where(self.neu_ident==self.neu_uniq[i])[0]
            t_spk_neu=self.spikes_raw[ind_spk_neu]
            for ii in range(self.num_trials):
                tstr=(self.rwin_time[ii]+self.start_window)
                for iii in range(self.num_steps):
                    lower=(tstr+iii*self.resol)
                    upper=(tstr+(iii+1)*self.resol)
                    rate[ii,i,iii]=len(t_spk_neu[(t_spk_neu>=lower)&(t_spk_neu<upper)])/float(self.resol)
        return rate

#     def remove_low(self,rate_pre,threshold):
#         thres=0
#         mean_neu=np.mean(np.sum(self.resol*rate_pre,axis=2)/(self.end_window+2.0),axis=0)
#         ind_in=np.where(mean_neu>threshold)[0]
#         rate=rate_pre[:,ind_in]
#         return rate

#     def normalize_all(self,spikes_pre):
#         spikes_mean=np.mean(spikes_pre)
#         spikes_std=np.std(spikes_pre)
#         spikes=(spikes_pre-spikes_mean)/spikes_std
#         return spikes

        
#     def encoding_model(self,kernel_size):
#         # Scacamos primero rates en funcion del tiempo para cada trial y neurona 
#         rate_pre=nan*np.zeros((self.num_trials,len(self.neu_uniq),self.num_steps))
#         for i in range(len(self.neu_uniq)):
#             ind_spk_neu=np.where(self.neu_ident==self.neu_uniq[i])[0]
#             t_spk_neu=self.spikes_raw[ind_spk_neu]
#             for ii in range(self.num_trials):
#                 tstr=(self.rwin_time[ii]-2.0)
#                 for iii in range(self.num_steps):
#                     lower=(tstr+iii*self.resol)
#                     upper=(tstr+(iii+1)*self.resol)
#                     rate_pre[ii,i,iii]=len(t_spk_neu[(t_spk_neu>lower)&(t_spk_neu<upper)])/float(self.resol)
#         #
#         num_steps_pertrial_kernel=int(round((2+self.end_window-kernel_size)/self.resol)+1)
#         num_steps_kernel_uses=int(round(kernel_size/self.resol))
#         rate=nan*np.zeros((len(self.neu_uniq),self.num_trials*num_steps_pertrial_kernel))
#         for i in range(len(self.neu_uniq)):
#             rate_i=rate_pre[:,i,(num_steps_kernel_uses-1):]
#             rate_i_reshape=np.reshape(rate_i,(self.num_trials*num_steps_pertrial_kernel))
#             rate[i]=rate_i_reshape
        return rate

