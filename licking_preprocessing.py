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

class licking():
    def __init__(self,reference_time,licking_time,lick_side,dic_time): 
        self.ref_time=reference_time
        self.licking_time=licking_time
        self.lick_side=lick_side
        self.num_trials=len(self.ref_time)
        #
        self.end_window=dic_time['end_window']
        self.start_window=dic_time['start_window']
        self.resol=dic_time['resol']
        self.num_steps=int(np.round((self.end_window-self.start_window)/self.resol))
        
    def rate_time(self):
        rate=nan*np.zeros((self.num_trials,self.num_steps))
        for ii in range(self.num_trials):
            tl=(self.ref_time[ii]+self.start_window)
            for iii in range(self.num_steps):
                lower=(tl+iii*self.resol)
                upper=(tl+(iii+1)*self.resol)
                li=self.lick_side[(self.licking_time>lower)&(self.licking_time<upper)]
                rate[ii,iii]=np.sum(li)/float(self.resol)
        return rate

    # def zeros_unos_munos(self): # Lo ideal es poner aqui la resolucion cuanto mas baja mejor
    #     binary=nan*np.zeros((self.num_trials,self.num_steps))
    #     for ii in range(self.num_trials):
    #         tstr=(self.rwin_time[ii]-2.0)
    #         for iii in range(self.num_steps):
    #             lower=(tstr+iii*self.resol)
    #             upper=(tstr+(iii+1)*self.resol)
    #             li=self.lick_side[(self.licking_time>lower)&(self.licking_time<upper)]
    #             if len(li)==0:
    #                 binary[ii,iii]=0.0
    #             if len(li)!=0:
    #                 binary[ii,iii]=np.sign(np.mean(li))
    #     return binary

    # def encoding_model_dependent(self,kernel_size):
    #     # Scacamos primero rates en funcion del tiempo para cada trial y neurona 
    #     rate_pre=nan*np.zeros((self.num_trials,self.num_steps))
    #     for ii in range(self.num_trials):
    #         tstr=(self.rwin_time[ii]-2.0)
    #         for iii in range(self.num_steps):
    #             lower=(tstr+iii*self.resol)
    #             upper=(tstr+(iii+1)*self.resol)
    #             li=self.lick_side[(self.licking_time>lower)&(self.licking_time<upper)]
    #             rate_pre[ii,iii]=np.sum(li)/float(self.resol)
    #     num_steps_pertrial_kernel=int(round((2+self.end_window-kernel_size)/self.resol)+1)
    #     num_steps_kernel_uses=int(round(kernel_size/self.resol))
    #     rate_t=rate_pre[:,(num_steps_kernel_uses-1):]
    #     rate=np.reshape(rate_t,(self.num_trials*num_steps_pertrial_kernel))
    #     return rate

    # def encoding_model_independent(self,kernel_size):
    #     # Scacamos primero rates en funcion del tiempo para cada trial y neurona 
    #     rate_pre=nan*np.zeros((self.num_trials,self.num_steps))
    #     #print self.licking_time
    #     for ii in range(self.num_trials):
    #         tstr=(self.rwin_time[ii]-2.0)
    #         for iii in range(self.num_steps):
    #             lower=(tstr+iii*self.resol)
    #             upper=(tstr+(iii+1)*self.resol)
    #             li=self.lick_side[(self.licking_time>lower)&(self.licking_time<upper)]
    #             rate_pre[ii,iii]=np.sum(li)/float(self.resol)

    #     # Hacemos aqui un reshape
    #     num_steps_pertrial_kernel=int(round((2+self.end_window-kernel_size)/self.resol)+1)
    #     num_steps_kernel_uses=int(round(kernel_size/self.resol))
    #     rate=nan*np.zeros((self.num_trials*num_steps_pertrial_kernel,num_steps_kernel_uses))
    #     gg=0
    #     for i in range(self.num_trials):
    #         rate_time=rate_pre[i]
    #         for ii in range(num_steps_pertrial_kernel):
    #             rate[gg]=rate_time[ii:(ii+num_steps_kernel_uses)]
    #             gg=gg+1
        return rate


        
    
