import os
#import matplotlib.pylab as plt
import numpy as np
import scipy
#import ipdb
import math
import sys
#import tables
import sys
from scipy.stats import sem
from scipy.stats import pearsonr
nan=float('nan')

class features_whiskers_contacts():
    def __init__(self,start_frame,rwin_frame,stop_frame,cont_str_frame,quantity,whis_ident,time_lock,start_window,end_window,resol): #cont_end_frame,
        self.start_frame=start_frame
        self.stop_frame=stop_frame
        self.rwin_frame=rwin_frame
        
        self.cont_str_frame=cont_str_frame
        #self.cont_end_frame=cont_end_frame # do we need this?
        self.quantity=quantity
        self.whis_ident=whis_ident
        #self.whis_uniq=np.unique(self.whis_ident)
        self.whis_uniq=np.array(['C0','C1','C2','C3']) # This is to prevent error with sessions where there are no C0 contacts
        self.num_whisk=len(self.whis_uniq)
        self.num_trials=len(self.rwin_frame)
        #
        self.start_window_sec=start_window
        self.end_window_sec=end_window
        self.resol_sec=resol
        
        self.time_lock=time_lock
        self.start_window_frame=int(self.start_window_sec*200)
        self.end_window_frame=int(self.end_window_sec*200)
        self.resol_frame=int(self.resol_sec*200)
        self.num_steps=int(np.round((self.end_window_frame-self.start_window_frame)/self.resol_frame))
        
    def convolutional(self):
        if len(self.quantity)==1:
            feat=np.zeros((self.num_trials,1,len(self.whis_uniq),self.num_steps))
        else:
            feat=nan*np.zeros((self.num_trials,1,len(self.whis_uniq),self.num_steps))
        #
        if self.time_lock=='response':
            tl_vec=self.rwin_frame
        if self.time_lock=='stop_move':
            tl_vec=self.stop_frame
        #
        for i in range(len(self.whis_uniq)):
            ind_cont_whis=np.where(self.whis_ident==self.whis_uniq[i])[0]
            cont_str_whis=self.cont_str_frame[ind_cont_whis]
            #cont_end_whis=self.cont_end_frame[ind_cont_whis]
            if len(self.quantity)>1:
                quantity_whis=self.quantity[ind_cont_whis]
            for ii in range(self.num_trials):
                tl=(tl_vec[ii]+self.start_window_frame)
                for iii in range(self.num_steps):
                    lower=(tl+iii*self.resol_frame)
                    upper=(tl+(iii+1)*self.resol_frame)
                    index_final=(cont_str_whis>=lower)&(cont_str_whis<upper)
                    if len(self.quantity)==1:
                        feat[ii,0,i,iii]=len(cont_str_whis[index_final])/(self.resol_sec)
                    else:
                        if np.sum(index_final)!=0:
                            feat[ii,0,i,iii]=np.mean(quantity_whis[index_final])
        return feat

class features_whiskers_all_frames(): 
    def __init__(self,start_frame,rwin_frame,stop_frame,quantity,frames,whis_ident,time_lock,start_window,end_window,resol):
        self.start_frame=start_frame
        self.stop_frame=stop_frame
        self.rwin_frame=rwin_frame
        self.quantity=quantity
        self.frames=frames
        self.whis_ident=whis_ident
        self.whis_uniq=np.unique(self.whis_ident)
        self.num_whisk=len(self.whis_uniq)
        self.num_trials=len(self.rwin_frame)
        #
        self.start_window_sec=start_window
        self.end_window_sec=end_window
        self.resol_sec=resol
        
        self.time_lock=time_lock
        self.start_window_frame=int(self.start_window_sec*200)
        self.end_window_frame=int(self.end_window_sec*200)
        self.resol_frame=int(self.resol_sec*200)
        self.num_steps=int((self.end_window_frame-self.start_window_frame)/self.resol_frame)

    def convolutional(self):
        feat=nan*np.zeros((self.num_trials,1,len(self.whis_uniq),self.num_steps))
        if self.time_lock=='response':
            tl_vec=self.rwin_frame
        if self.time_lock=='stop_move':
            tl_vec=self.stop_frame
        i=0
        while i<len(self.whis_uniq):
            ind_whis=(self.whis_ident==self.whis_uniq[i])
            quantity_whis=self.quantity[ind_whis]
            frames_whis=self.frames[ind_whis]
            ii=0
            while ii<self.num_trials:
                tl=(tl_vec[ii]+self.start_window_frame)
                iii=0
                while iii<self.num_steps:
                    lower=(tl+iii*self.resol_frame)
                    upper=(tl+(iii+1)*self.resol_frame)
                    ind_step=((frames_whis>lower)*(frames_whis<upper))
                    if any(ind_step):
                        feat[ii,0,i,iii]=np.mean(quantity_whis[ind_step])
                    iii=iii+1
                ii=ii+1
            i=i+1
        return feat

 
