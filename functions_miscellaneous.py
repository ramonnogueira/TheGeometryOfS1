import os
import matplotlib.pylab as plt
import numpy as np
import scipy
#import ipdb
import math
import sys
#import tables
import pandas
import pickle as pkl
#import torch
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ortho_group 
import itertools
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sympy.utilities.iterables import multiset_permutations
from itertools import combinations

import features_whiskers
import licking_preprocessing
import spikes_processing
import nn_pytorch_encoders
#import pytorch_nonlinear_random_mapping

nan=float('nan')

##################################################
# Functions to read lick files

def read_logfile_into_df(logfile, nargs=4, add_trial_column=False,unsorted_times_action='warn'):
    """Read logfile into a DataFrame
    
    Something like this should probably be the preferred way to read the 
    lines into a structured data frame.
    
    Use get_commands_from_parsed_lines to parse the arguments, eg,
    converting to numbers.
    
    Each line in the file will be a row in the data frame.
    Each line is separated by whitespace into the different columns.
    Thus, the first column will be "time", the second "command", and the
    rest "arguments".
    
    nargs : how many argument columns to add. Lines that contain more arguments
        than this will be silently truncated! Lines with fewer will be padded
        with None.
    add_trial_column : optionally add a column for the trial number of 
        each line. Lines before the first trial begins have trial number -1.
    
    unsorted_times_action : 'ignore', 'warn', 'error'
        The times "should" be sorted but frequently aren't.
        This is always the case for some commands, and less frequently
        the case when the first digit seems to have been dropped.
    
    The dtype will always be int for the time column and object (ie, string)
    for every other column. This is to ensure consistency. You may want
    to coerce certain columns into numeric dtypes.
    """
    # Determine how many argument columns to use
    arg_cols = ['arg%d' % n for n in range(nargs)]
    all_cols = ['time', 'command'] + arg_cols
    
    # Set dtypes
    dtype_d = {'time': np.int, 'command': np.object}
    for col in arg_cols:
        dtype_d[col] = np.object
    
    # Read. Important to avoid reading header of index or you can get
    # weird errors here, like unnamed columns.
    rdf = pandas.read_csv(logfile, sep=' ', names=all_cols, 
        index_col=False, header=None)
    if not np.all(rdf.columns == all_cols):
        raise IOError("cannot read columns correctly from logfile")
    
    # Convert time to integer and drop malformed lines
    #new_time = rdf['time'].convert_objects(convert_numeric=True)
    new_time = pandas.to_numeric(rdf['time'], errors='coerce')
    if new_time.isnull().any():
        print ("warning: malformed time string at line(s) %r") % (
            new_time.index[new_time.isnull()].values)
    rdf['time'] = new_time
    rdf = rdf.loc[~rdf['time'].isnull()]
    rdf['time'] = rdf['time'].astype(np.int)
    
    # Convert dtypes. We have to do it here, because if done during reading
    # it will crash on mal-formed dtypes. Could catch that error and then
    # run this...
    # Well this isn't that useful because it leaves dtypes messed up. Need
    # to find and drop the problematic lines.
    for col, dtyp in dtype_d.items():
        try:
            rdf[col] = rdf[col].astype(dtyp)
        except ValueError:
            raise IOError("cannot coerce %s to %r" % (col, dtyp))
    
    # Join on trial number
    if add_trial_column:
        # Find the boundaries between trials in logfile_lines
        trl_start_idxs = rdf.index[rdf['command'] == start_trial_token]

        if len(trl_start_idxs) > 0:            
            # Assign trial numbers. The first chunk of lines are 
            # pre-session setup, so subtract 1 to make that trial "-1".
            # Use side = 'right' to place TRL_START itself correctly
            rdf['trial'] = np.searchsorted(np.asarray(trl_start_idxs), 
                np.asarray(rdf.index), side='right') - 1        
    
    # Error check
    # Very commonly the ACK TRL_RELEASED, SENH, AAR_L, and AAR_R commands
    # are out of order. So ignore this for now.
    # Somewhat commonly, there is a missing first digit of the time, for
    # some reason.
    rrdf = rdf[
        ~rdf.command.isin(['DBG', 'ACK', 'SENH']) &
        ~rdf.arg0.isin(['AAR_L', 'AAR_R'])
        ]
    unsorted_times = rrdf['time'].values
    bad_args = np.where(np.diff(unsorted_times) < 0)[0]
    if len(bad_args) > 0:
        first_bad_arg = bad_args[0]
        pre_bad_arg = np.max([first_bad_arg - 2, 0])
        post_bad_arg = np.min([first_bad_arg + 2, len(rrdf)])
        bad_rows = rrdf.loc[rrdf.index[pre_bad_arg]:rrdf.index[post_bad_arg]]
        error_string = "unsorted times in logfile %s, starting at line %d" % (
            logfile, bad_args[0])

        if unsorted_times_action == 'warn':
            print (error_string)
        elif unsorted_times_action == 'error':
            raise ValueError(error_string)
        elif unsorted_times_action == 'ignore':
            pass
        else:
            raise ValueError("unknown action for unsorted times")
    
    return rdf


def get_commands_from_parsed_lines(parsed_lines, command,
    arg2dtype=None):
    """Return only those lines that match "command" and set dtypes.
    
    parsed_lines : result of read_logfile_into_df
    command : 'ST_CHG', 'ST_CHG2', 'TCH', etc.
        This is used to select rows from parsed_lines. For known arguments,
        we can also use this to set arg2dtype.
    arg2dtype : dict explaining which args to keep and what dtype to convert
        e.g., {'arg0': np.int, 'arg1': np.float}
    
    Returns:
        DataFrame with one row for each matching command, and just the
        requested columns. We always include 'time', 'command', and
        'trial' if available
    
    Can use something like this to group the result by trial and arg0:
    tt2licks = lick_times.groupby(['trial', 'arg0']).groups
    for (trial, lick_type) in tt2licks:
        tt2licks[(trial, lick_type)] = \
            ldf.loc[tt2licks[(trial, lick_type)], 'time'].values / 1000.    
    
    See BeWatch.misc for other examples of task-specific logic
    """
    # Pick
    res = parsed_lines[parsed_lines['command'] == command]
    
    # Decide which columns to keep and how to coerce
    if command == 'ST_CHG2':
        if arg2dtype is None:
            arg2dtype = {'arg0': np.int, 'arg1': np.int}
    elif command == 'ST_CHG':
        if arg2dtype is None:
            arg2dtype = {'arg0': np.int, 'arg1': np.int}
    elif command == 'TCH':
        arg2dtype = {'arg0': np.int}
    
    if arg2dtype is None:
        raise ValueError("must provide arg2dtype")
    
    # Keep only the columns we want
    keep_cols = ['time', 'command'] + sorted(arg2dtype.keys())
    if 'trial' in res.columns:
        keep_cols.append('trial')    
    res = res[keep_cols]

    # Coerce dtypes
    for argname, dtyp in arg2dtype.items():
        try:
            res[argname] = res[argname].astype(dtyp)
        except ValueError:
            print ("warning: cannot coerce column %s to %r") % (argname, dtyp)

    return res

#########################################################################################
# Nice figures 

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines: 
            if loc=='left':
                spine.set_position(('outward', 10))  # outward by 10 points
            if loc=='bottom':
                spine.set_position(('outward', 0))  # outward by 10 points
         #   spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        
##########################################################################################
# Functions to extract files and order them 

def extract_opto(abs_path):
    files_df=pandas.read_pickle(os.path.join(abs_path,'20200414_include_session_df'))
    files_labels_df=files_df.index.values.tolist()
    opto_dic={}
    for i in files_labels_df:
        opto_dic[i[2]]={}
        opto_dic[i[2]]['opto']=files_df.loc[i]['opto']
        opto_dic[i[2]]['sham']=files_df.loc[i]['sham']
    return opto_dic

def extract_contacts(abs_path):
    dic={}
    col_cont_summ=pandas.read_pickle(os.path.join(abs_path,'colorized_contacts_summary'))
    dic['contact_start']=np.array(col_cont_summ['frame_start'],dtype=np.int32)
    dic['contact_end']=np.array(col_cont_summ['frame_stop'],dtype=np.int32)
    dic['fol_x']=np.array(col_cont_summ['fol_x'],dtype=np.float64)
    dic['fol_y']=np.array(col_cont_summ['fol_y'],dtype=np.float64)
    dic['tip_x']=np.array(col_cont_summ['tip_x'],dtype=np.float64)
    dic['tip_y']=np.array(col_cont_summ['tip_y'],dtype=np.float64)
    dic['angle']=np.array(col_cont_summ['angle'],dtype=np.float64)
    dic['whisker']=np.array(col_cont_summ['whisker'])
    return dic

def extract_analog(abs_path):
    dic={}
    col_whis_ends=pandas.read_pickle(os.path.join(abs_path,'colorized_whisker_ends'))
    dic['frames']=np.array(col_whis_ends['frame'],dtype=np.int32)
    dic['angle']=np.array(col_whis_ends['angle'],dtype=np.float64)
    dic['tip_x']=np.array(col_whis_ends['tip_x'],dtype=np.float64)
    dic['tip_y']=np.array(col_whis_ends['tip_y'],dtype=np.float64)
    dic['fol_x']=np.array(col_whis_ends['fol_x'],dtype=np.float64)
    dic['fol_y']=np.array(col_whis_ends['fol_y'],dtype=np.float64)
    dic['whisker']=np.array(col_whis_ends['whisker'])
    return dic

def extract_licks(abs_path):
    dic={}
    # Data licking in original timebase
    ldf=read_logfile_into_df(os.path.join(abs_path,'behavioral_logfile'))
    lick_times_prepre=get_commands_from_parsed_lines(ldf, 'TCH')        
    lick_times_pre=lick_times_prepre[lick_times_prepre.arg0.isin([1, 2])]
    licking_time=np.array(lick_times_pre['time'],dtype=np.float64)/1000
    lick_side=np.array(lick_times_pre['arg0'],dtype=np.int32)
    lick_side[lick_side==2]=-1 # 1 is left, -1 is right
    dic['licking_time']=licking_time
    dic['lick_side']=lick_side
    return dic

def extract_index_use(opto,abs_path): # Removing trials: last, no random stim presentation, missed, reward spoil, all activation opto
    data_behav=pandas.read_pickle(os.path.join(abs_path,'trial_matrix'))
    #
    index_use_rnd=np.array(data_behav['isrnd'])
    index_use_rnd[-1]=False
    reward_pre=data_behav['outcome']
    choice_pre=data_behav['choice']
    index_use_rw=np.array(reward_pre!='spoil')
    index_use_ch=np.array(choice_pre!='nogo')
    opto_trials=np.array(data_behav['opto']) # 3 laser on; 2 laser off
    if opto['opto']=='True':
        index_use_opto=np.array(opto_trials==2)
    else:
        index_use_opto=np.ones(len(opto_trials),dtype=bool)
    index_use=(index_use_rnd*index_use_rw*index_use_ch*index_use_opto)
    return index_use
    
def extract_behavior(index_use,index_lick,abs_path):
    dic={}
    data_behav=pandas.read_pickle(os.path.join(abs_path,'trial_matrix'))
    # reward
    reward=data_behav['outcome'][index_use][index_lick]
    reward[reward=='hit']=1
    reward[reward=='error']=-1
    reward=np.array(reward,dtype=np.int8)
    dic['reward']=reward
    # choice
    choice=data_behav['choice'][index_use][index_lick]
    choice[choice=='left']=1
    choice[choice=='right']=-1
    choice=np.array(choice,dtype=np.int8)
    dic['choice']=choice
    # stimulus. +1 is concave, -1 is convex
    stimulus=data_behav['rewside'][index_use][index_lick]
    stimulus[stimulus=='left']=1
    stimulus[stimulus=='right']=-1
    stimulus=np.array(stimulus,dtype=np.int8)
    dic['stimulus']=stimulus
    # stimulus position
    position=data_behav['servo_pos'][index_use][index_lick]
    position=np.array(position,dtype=np.float64)
    position_uniq=np.unique(position)
    position_double=(position*stimulus)
    pos_doub_uniq=np.unique(position_double)
    dic['position']=position
    # stimulus difficulty
    difficulty=data_behav['stepper_pos'][index_use][index_lick] # 199 and 150 stimulus left (+1), 100 and 50 stimulus right (-1)
    difficulty=np.array(difficulty,dtype=np.int8)
    dic['difficulty']=difficulty
    return dic

def extract_timings(index_use,abs_path):
    dic={}
    trial_timings=pandas.read_pickle(os.path.join(abs_path,'trial_matrix'))
    start_frame=np.array(trial_timings['exact_start_frame'],dtype=np.int32)[index_use]
    rwin_frame=np.array(trial_timings['rwin_frame'],dtype=np.int32)[index_use]    
    rwin_sec=np.array(trial_timings['rwin_time'],dtype=np.float64)[index_use]
    stop_frame=np.array(trial_timings['shape_stop_frame'],dtype=np.float32)[index_use]
    dic['start_frame']=start_frame
    dic['stop_frame']=stop_frame
    dic['rwin_frame']=rwin_frame
    dic['rwin_sec']=rwin_sec
    return dic

def extract_neural_timings(index_use,abs_path):
    dic={}
    neural_trial_timings=pandas.read_pickle(os.path.join(abs_path,'neural_trial_timings'))
    rwin_nbase=np.array(neural_trial_timings['rwin_time_nbase'],dtype=np.float64)[index_use]
    start_nbase=np.array(neural_trial_timings['start_time_nbase'],dtype=np.float64)[index_use]
    dic['rwin_nbase']=rwin_nbase
    dic['start_nbase']=start_nbase
    return dic

def extract_spikes(abs_path):
    dic={}
    spikes=pandas.read_pickle(os.path.join(abs_path,'spikes'))
    time=np.array(spikes['time'],dtype=np.float64)
    cluster=np.array(spikes['cluster'],dtype=np.float64)
    dic['time_stamps']=time
    dic['cluster']=cluster
    return dic

def extract_info_neu(files,abs_path,arx):
    dic={}
    neurons_df=pandas.read_pickle(os.path.join(abs_path,arx))
    all_neu_layer=np.array([])
    all_neu_inh=np.array([])
    for ii in range(len(files)):
        all_neu_layer=np.concatenate((all_neu_layer,np.array(neurons_df.loc[files[ii]]['layer'])))
        all_neu_inh=np.concatenate((all_neu_inh,np.array(neurons_df.loc[files[ii]]['NS'])))
    dic['all_neu_layer']=all_neu_layer
    dic['all_neu_inh']=all_neu_inh
    return dic

def extract_electrode_location(files,abs_path,arx):
    dic={}
    location_df=pandas.read_csv(os.path.join(abs_path,arx))
    neurons_df=pandas.read_pickle(os.path.join(abs_path,'20200414_big_waveform_info_df'))
    location_val=location_df.values
    dic_g={} #Location general
    dic_c={} #Location closest to C
    for i in range(len(location_val)):
        dic_g[location_val[i][0]]=location_val[i][2]
        dic_c[location_val[i][0]]=location_val[i][3]
    #
    location_general=np.array([])
    location_c=np.array([])
    for ii in range(len(files)):
        num_neu=len(neurons_df.loc[files[ii]]['layer'])
        arr_g=np.array([dic_g[files[ii]] for j in range(num_neu)])
        arr_c=np.array([dic_c[files[ii]] for j in range(num_neu)])
        location_general=np.concatenate((location_general,arr_g))
        location_c=np.concatenate((location_c,arr_c))
    dic['location_general']=location_general
    dic['location_c']=location_c
    return dic


def extract_whisk_cycles(abs_path):
    dic={}
    whisk_df=pandas.read_pickle(os.path.join(abs_path,'big_C2_tip_whisk_cycles'))
    all_neu=np.array([])
    for ii in range(len(files)):
        dic['layer_%s'%files[ii]]=np.array(neurons_df.loc[files[ii]]['layer'])
        dic['inh_%s'%files[ii]]=np.array(neurons_df.loc[files[ii]]['NS'])
        all_neu=np.concatenate((all_neu,np.array(neurons_df.loc[files[ii]]['layer'])))
    dic['all_neu']=all_neu
    dic['all_unique']=np.unique(all_neu)
    dic['num_total']=len(all_neu)
    return dic

################################################################################
# Create features

def create_feat(quantities,timings,analog,contact,dic_time):
    feat_all_ct=np.array([])
    feat_all_an=np.array([])
    quant_ct=quantities['contacts']
    quant_analog=quantities['analog']
    # Features contacts
    for i in range(len(quant_ct)):
        if quant_ct[i]=='contacts':
            feat_cl_ct=features_whiskers.features_whiskers_contacts(start_frame=timings['start_frame'],rwin_frame=timings['rwin_frame'],stop_frame=timings['stop_frame'],cont_str_frame=contact['contact_start'],quantity=[0],whis_ident=contact['whisker'],time_lock=dic_time['time_lock'],start_window=dic_time['start_window'],end_window=dic_time['end_window'],resol=dic_time['resol'])
        else:
            feat_cl_ct=features_whiskers.features_whiskers_contacts(start_frame=timings['start_frame'],rwin_frame=timings['rwin_frame'],stop_frame=timings['stop_frame'],cont_str_frame=contact['contact_start'],quantity=contact[quant_ct[i]],whis_ident=contact['whisker'],time_lock=dic_time['time_lock'],start_window=dic_time['start_window'],end_window=dic_time['end_window'],resol=dic_time['resol'])
        feat_ct=feat_cl_ct.convolutional()
        if i==0:
            feat_all_ct=feat_ct.copy()
        else:
            feat_all_ct=np.concatenate((feat_all_ct,feat_ct),axis=2)

    # Features analog
    for i in range(len(quant_analog)):
        feat_cl_an=features_whiskers.features_whiskers_all_frames(start_frame=timings['start_frame'],rwin_frame=timings['rwin_frame'],stop_frame=timings['stop_frame'],quantity=analog[quant_analog[i]],frames=analog['frames'],whis_ident=analog['whisker'],time_lock=dic_time['time_lock'],start_window=dic_time['start_window'],end_window=dic_time['end_window'],resol=dic_time['resol'])
        feat_an=feat_cl_an.convolutional()
        if i==0:
            feat_all_an=feat_an.copy()
        else:
            feat_all_an=np.concatenate((feat_all_an,feat_an),axis=2)

    if len(feat_all_ct)==0:
        feat_all=feat_all_an.copy()
    elif len(feat_all_an)==0:
        feat_all=feat_all_ct.copy()
    else:
        feat_all=np.concatenate((feat_all_ct,feat_all_an),axis=2)
    
    return feat_all

def create_licks(reference_time,licking_time,lick_side,dic_time):
    lick_pre=licking_preprocessing.licking(reference_time=reference_time,licking_time=licking_time,lick_side=lick_side,dic_time=dic_time)
    lick_rate=lick_pre.rate_time()
    return lick_rate

def create_firing_rate(rwin_time,spikes_raw,neu_ident,dic_time):
    fr_pre=spikes_processing.spikes(rwin_time=rwin_time,spikes_raw=spikes_raw,neu_ident=neu_ident,dic_time=dic_time)
    fr=fr_pre.rate_time()
    return fr

################################################################################################################
# Formats population activity, fast and slow features into num_trials x num_steps x num_neurons/features
# It removes the first trial (neurons) and last trial (features) (n_back)
# This format is very useful for the recurrent encoders. It will need further reformating for feedforward models
def crear_encoding_general(features_fast,features_slow,firing_rate,lick_rate,behavior): 
    num_trials=len(firing_rate)
    num_neurons=len(firing_rate[0])
    num_steps=len(firing_rate[0,0])
    n_back=1
    num_feats_fast=(len(features_fast[0]))
    num_feats_slow=(len(features_slow)) 
    
    # Firing rate. Devuelve tensor con num_trials x time_steps x num_neurons
    fr_encoding_pre=nan*np.zeros((num_trials,num_steps,num_neurons))
    for i in range(num_trials):
        fr_encoding_pre[i]=np.transpose(firing_rate[i])
    fr_encoding=fr_encoding_pre[n_back:]
    
    # Lick features. Devuelve tensor con num_trials x time_steps x 2
    lick_encoding_pre=nan*np.zeros((num_trials,num_steps,2))
    lick_encoding_pre[:,:,0]=lick_rate
    lick_encoding_pre[:,:,1]=abs(lick_rate)
    lick_encoding=lick_encoding_pre[n_back:]
           
    # Fast features. Devuelve tensor con num_trials x time_steps x num_feats_fast
    features_fast_encoding_pre=nan*np.zeros((num_trials,num_steps,num_feats_fast))
    for i in range(num_trials):
        features_fast_encoding_pre[i]=np.transpose(features_fast[i])
    features_fast_encoding=features_fast_encoding_pre[n_back:]

    # Reward
    reward=behavior['reward'][n_back:]
    # Stimulus
    stimulus=behavior['stimulus'][n_back:]
    
    # Fast slow. Devuelve tensor con num_trials x time_steps x (n_back+1)*num_feats_slow
    features_slow_encoding=nan*np.zeros((num_trials-n_back,num_steps,(n_back+1)*num_feats_slow))
    j=0
    for jj in range(num_feats_slow):
        quant=behavior[features_slow[jj]]
        for i in range(n_back+1):
            for ii in range(num_steps):
                features_slow_encoding[:,ii,j]=quant[i:(len(quant)-n_back+i)]
            j=(j+1)
                          
    features_encoding_all=np.concatenate((lick_encoding,features_fast_encoding,features_slow_encoding),axis=2) 
    
    dic={}
    dic['firing_rate']=fr_encoding
    dic['features']=features_encoding_all
    dic['reward']=reward
    dic['stimulus']=stimulus
    return dic

# Takes the data formated as in "crear_encoding_general" and puts it into the encoding format
# for feedforward networks
def crear_encoding_feedforward(firing_rate_general,features_general,features_slow,dic_time,reward):
    num_trials=len(firing_rate_general)
    num_steps_l=len(firing_rate_general[0])
    num_neurons=len(firing_rate_general[0,0])
    #
    n_back=1
    wind_start=(int(dic_time['size_kernel']/dic_time['resol'])-1)
    num_steps_kernel=int(dic_time['size_kernel']/dic_time['resol'])
    num_steps_s=(num_steps_l-wind_start)
    #
    num_feats_total=len(features_general[0,0])
    num_feats_slow=((n_back+1)*len(features_slow)) 
    num_feats_fast=(num_feats_total-num_feats_slow)

    #############################################################
    # Firing rate. Returns matrix shape num_trials*num_steps_s x num_neurons
    fr_encoding=firing_rate_general[:,wind_start:].reshape(-1,num_neurons)

    ############################################################################################################
    # Fast features. Devuelve una matriz con num_trials*num_steps x num_steps_kernel*(num_feats_fast +2 licking)
    features_fast_pre=features_general[:,:,0:num_feats_fast]
    features_fastlick_encoding=nan*np.zeros((num_trials*num_steps_s,num_steps_kernel*num_feats_fast)) 
    for i in range(num_trials):                                                                                                                                                                           
        for ii in range(num_steps_s):
            for iii in range(num_feats_fast):                                                                                                                                                             
                features_fastlick_encoding[(i*num_steps_s+ii),iii*num_steps_kernel:(iii+1)*num_steps_kernel]=features_fast_pre[i,ii:(ii+num_steps_kernel),iii]   

    ####################################################################################
    # Slow features. 
    # Shape matrix  num_trials x num_feats_slow*num_steps_s x num_steps_s
    features_general_slow=features_general[:,0,-num_feats_slow:]
    features_slow_pre=nan*np.zeros((num_trials,num_feats_slow*num_steps_s,num_steps_s))
    for j in range(num_trials):
        b0=nan*np.zeros((num_feats_slow*num_steps_s,num_steps_s))
        for jj in range(num_feats_slow):
            b1=np.zeros((num_steps_s,num_steps_s))
            b1[np.diag_indices(num_steps_s)]=features_general_slow[j,jj]
            b0[jj*num_steps_s:(jj+1)*num_steps_s]=b1
        features_slow_pre[j]=b0
    num_feats_slow_all=(num_feats_slow*num_steps_s)
    
    # Slow features. Returns matrix shape num_trials*num_steps_s x num_feats_slow_all
    features_slow_encoding=nan*np.zeros((num_trials*num_steps_s,num_feats_slow_all))
    for i in range(num_trials):
        for ii in range(num_steps_s):
            for iii in range(num_feats_slow_all):
                features_slow_encoding[(i*num_steps_s+ii),iii]=features_slow_pre[i,iii,ii]

    features_encoding_all=np.concatenate((features_fastlick_encoding,features_slow_encoding),axis=1)

    # Reward. Returns matrix shape num_trials*num_steps_s 
    reward_encoding=nan*np.zeros((num_trials*num_steps_s))
    for i in range(num_trials):
        for ii in range(num_steps_s):
            reward_encoding[(i*num_steps_s+ii)]=reward[i]
    
    dic={}
    dic['firing_rate']=fr_encoding
    dic['features']=features_encoding_all
    dic['reward']=reward_encoding
    return dic

# Returns stimulus in encoding format. It should be given stimulus in already the "general" format
def stimulus_encoding_format(num_trials,num_steps_s,stimulus):
    stimulus_encoding=nan*np.zeros((num_trials*num_steps_s))
    for i in range(num_trials):
        for ii in range(num_steps_s):
            stimulus_encoding[(i*num_steps_s+ii)]=stimulus[i]
    return stimulus_encoding
            
# Takes the data formated as in "crear_encoding_feedforward" and undoes it to trials x timesteps x neurons format
def undo_encoding_feedforward(firing_rate,nt_real,n_steps):
    nt=len(firing_rate)
    num_neurons=len(firing_rate[0])
    
    # Firing rate. Returns matrix shape num_trials*num_steps_s x num_neurons
    fr_undo=np.zeros((nt_real,num_neurons,n_steps))
    for i in range(nt_real):
        fr_undo[i]=np.transpose(firing_rate[i*n_steps:(i+1)*n_steps])

    dic={}
    dic['firing_rate']=fr_undo
    return dic

def crear_encoding_feedforward_ANN(firing_rate,features,size_kernel):
    num_trials=len(firing_rate)
    num_neurons=len(firing_rate[0,0])
    num_feats=len(features[0,0])
    num_steps_pre=len(firing_rate[0])
    num_steps=(num_steps_pre-size_kernel+1)
    
    # Firing rate. Returns matrix shape num_trials*num_steps_s x num_neurons
    fr_encoding=firing_rate[:,(size_kernel-1):].reshape(-1,num_neurons)
    
    #Fast features. Devuelve una matriz con num_trials*num_steps x num_steps_kernel*num_feats_fast
    features_encoding=nan*np.zeros((num_trials*num_steps,size_kernel*num_feats))                                                                                                           
    for i in range(num_trials):
        for ii in range(num_steps):
            for iii in range(num_feats):                                                                                                                                                             
                features_encoding[(i*num_steps+ii),iii*size_kernel:(iii+1)*size_kernel]=features[i,ii:(ii+size_kernel),iii]            
    dic={}
    dic['firing_rate']=fr_encoding
    dic['features']=features_encoding
    return dic

###################################################################################

def normalize_feat_time_general(feat_pre):
    feat=nan*np.zeros(np.shape(feat_pre))
    for i in range(len(feat_pre[0,0])):
        feat_trans=feat_pre[:,:,i]
        feat_mean=np.nanmean(feat_trans)
        feat_std=np.nanstd(feat_trans)
        feat[:,:,i]=(feat_trans-feat_mean)/feat_std
        feat[:,:,i][np.isnan(feat[:,:,i])]=0.0
    return feat

def normalize_fr_time_general(fr):
    zscore=nan*np.zeros(np.shape(fr))
    num_neu=len(fr[0,0])
    for i in range(num_neu):
        neu_trans=fr[:,:,i]
        neu_mean=np.nanmean(neu_trans)
        neu_std=np.nanstd(neu_trans)
        zscore[:,:,i]=(neu_trans-neu_mean)/neu_std
    return zscore

def normalize_feat_time(feat_pre):
    feat=nan*np.zeros(np.shape(feat_pre))
    for i in range(len(feat_pre[0])):
        feat_trans=feat_pre[:,i]
        feat_mean=np.nanmean(feat_trans)
        feat_std=np.nanstd(feat_trans)
        feat[:,i]=(feat_trans-feat_mean)/feat_std
        feat[:,i][np.isnan(feat[:,i])]=0.0
    return feat

# def normalize_fr_time(fr_pre):
#     fr=nan*np.zeros(np.shape(fr_pre))
#     for i in range(len(fr_pre[0])):
#         fr_trans=fr_pre[:,i]
#         fr_mean=np.nanmean(fr_trans)
#         fr_std=np.nanstd(fr_trans)
#         fr[:,i]=(fr_trans-fr_mean)/fr_std
#         fr[:,i][np.isnan(fr[:,i])]=0.0
#     return fr

def normalize_feat_conv_time(feat_pre):
    feat=nan*np.zeros(np.shape(feat_pre))
    for i in range(len(feat_pre[0,0])):
        feat_trans=feat_pre[:,0,i]
        feat_mean=np.nanmean(feat_trans)
        feat_std=np.nanstd(feat_trans)
        feat[:,0,i]=(feat_trans-feat_mean)/feat_std
        feat[:,0,i][np.isnan(feat[:,0,i])]=0.0
    return feat

def normalize_lick_time(lick_pre):
    lick=nan*np.zeros(np.shape(lick_pre))
    lick_mean=np.nanmean(lick_pre)
    lick_std=np.nanstd(lick_pre)
    lick=(lick_pre-lick_mean)/lick_std
    lick[np.isnan(lick)]=0.0
    return lick


####################################################################
# Create all super

def create_super_mouse(all_super,behavior,feat):
    all_super_ret={}
    all_super_ret['stimulus']=np.concatenate((all_super['stimulus'],behavior['stimulus']))
    all_super_ret['choice']=np.concatenate((all_super['choice'],behavior['choice']))
    all_super_ret['reward']=np.concatenate((all_super['reward'],behavior['reward']))
    all_super_ret['position']=np.concatenate((all_super['position'],behavior['position']))
    all_super_ret['difficulty']=np.concatenate((all_super['difficulty'],behavior['difficulty']))
    all_super_ret['feat']=np.concatenate((all_super['feat'],feat))
    return all_super_ret

###############################################################################
# Equal number of correct and mistakes
def equal_corr_incorr(reward,n_decorr):
    ind_corr=np.where(reward==1)[0]
    ind_incorr=np.where(reward==-1)[0]
    index_def=np.zeros((n_decorr,2*len(ind_incorr)),dtype=np.int32)
    for yy in range(n_decorr):
        ind_corr_i=np.random.choice(ind_corr,len(ind_incorr),replace=False)
        index_def[yy]=np.sort(np.concatenate((ind_corr_i,ind_incorr)))
    return index_def

# bootstrap errobars
def boot_errorbars(data,n):
    n_trials=len(data)
    boot_data=nan*np.zeros((n,len(data[0])))
    for nn in range(n):
        ind_rnd=np.random.choice(np.arange(n_trials),n_trials,replace=True)
        boot_data[nn]=np.mean(data[ind_rnd],axis=0)
    return boot_data
        

#################################################################################
# SVM cv

def svm_reg(feat,clase,n_cv,reg,balanced):
    perf=np.zeros((2,n_cv))
    wei=np.zeros((n_cv,len(feat[0])))
    skf=StratifiedKFold(n_cv)
    g=0
    for train,test in skf.split(feat,clase):
        if balanced:
            mod=LinearSVC(dual=False,C=reg,class_weight='balanced')
        else:
            mod=LinearSVC(dual=False,C=reg)
        mod.fit(feat[train],clase[train])
        perf[0,g]=mod.score(feat[train],clase[train])
        perf[1,g]=mod.score(feat[test],clase[test])
        wei[g]=mod.coef_[0]
        g=g+1
    dic={}
    performance=np.mean(perf,axis=1)
    weights=np.mean(wei,axis=0)
    dic['performance']=performance
    dic['weights']=weights
    return dic
######################################################

# Creates the XOR task (N-dim Parity) based on splitting each feature on the median
# It needs to create a feature vector with i.i.d. gaussian noise in order to do a meaningful median split
# It creates balanced classes but a bit unrealistic because of iid gaussian noise
# Almost imposible for linear, very complicated for nonlinear
def create_xor_median(feat,clase,n_used,std,index_feat_xor,index_t_xor):
    reg_vec=np.logspace(-5,5,10)
    n_val=4
    n_cv=4
    #
    # IMPORTANT: Feat has shape of (n_trials , n_steps , n_feat) ("encoding general")
    feat_noise_all=(feat+np.random.normal(0,std,(np.shape(feat))))
    feat_noise=feat_noise_all[:,index_t_xor][:,:,index_feat_xor]
    nf_pre=len(feat_noise[0,0])
    nt_pre=len(feat_noise[0])
    n_feat=nf_pre*nt_pre
    feat_noise_flat=(np.reshape(feat_noise,(len(feat),n_feat)))
    
    feat_binary=nan*np.zeros(np.shape(feat_noise_flat))
    info_feat=nan*np.zeros(n_feat)
    len_class=nan*np.zeros((2,n_feat))
    for j in range(n_feat):
        svm_cl=svm_sum_val(feat=feat_noise_flat[:,(j):(j+1)],clase=clase,reg_vec=reg_vec,n_val=n_val,n_cv=n_cv)
        svm=svm_cl.svm()
        perf_test=svm['performance'][1]
        index_max=np.argmax(perf_test)
        info_feat[j]=svm['performance'][2][index_max]

        median=np.median(feat_noise_flat[:,j])
        feat_binary[:,j][feat_noise_flat[:,j]>median]=1
        feat_binary[:,j][feat_noise_flat[:,j]<=median]=0
        len_class[0,j]=len(feat_binary[:,j][feat_binary[:,j]==1])
        len_class[1,j]=len(feat_binary[:,j][feat_binary[:,j]==0])

    feat_used=np.argsort(info_feat)[-n_used:]
    xor=np.sum(feat_binary[:,feat_used],axis=1)%2

    dic={}
    dic['feat_noise_all']=feat_noise_all
    dic['feat_xor_noise']=feat_noise_flat[:,feat_used]
    dic['feat_binary']=feat_binary[:,feat_used]
    dic['feat_used']=feat_used
    dic['len_class']=len_class
    dic['xor']=xor
    return dic

# Same as before but with the encoding format
def create_xor_median_encoding(feat,v_used,std):
    feat_noise_all=(feat+np.random.normal(0,std,(np.shape(feat))))
    feat_noise=feat_noise_all[:,v_used]
    n_feat=len(v_used)

    m_orto=ortho_group.rvs(dim=n_feat)
    feat_trans=np.dot(feat_noise,m_orto)
    
    feat_binary=nan*np.zeros(np.shape(feat_trans))
    feat_binary_orig=nan*np.zeros(np.shape(feat_trans))
    for j in range(n_feat):
        #print (j)
        # Rotated Space
        median=np.median(feat_trans[:,j])
        #median=np.mean(feat_trans[:,j]) # Cuidado!!!!
        feat_binary[:,j][feat_trans[:,j]>median]=1
        feat_binary[:,j][feat_trans[:,j]<=median]=0
        # Original Space
        median_orig=np.median(feat_noise[:,j])
        #median_orig=np.mean(feat_noise[:,j]) # Cuidado!!!!
        feat_binary_orig[:,j][feat_noise[:,j]>median_orig]=1
        feat_binary_orig[:,j][feat_noise[:,j]<=median_orig]=0
    xor=np.sum(feat_binary,axis=1)%2
    lin=feat_binary[:,0]

    dic={}
    dic['feat_noise_all']=feat_noise_all
    dic['feat_noise']=feat_noise
    dic['feat_trans']=feat_trans
    dic['feat_binary']=feat_binary
    dic['feat_binary_orig']=feat_binary_orig
    dic['xor']=xor
    dic['lin']=lin
    return dic

def create_xor_median_encoding_new(feat,v_used,std):
    n_feat=len(v_used)
    n_opt=20
    xor_vec=nan*np.zeros(n_opt)
    orto_vec=nan*np.zeros((n_opt,n_feat,n_feat))
    # We optimize for the rotation with the lowest linear performance
    for tt in range(n_opt):
        feat_noise_all=(feat+np.random.normal(0,std,(np.shape(feat))))
        feat_noise=nan*np.zeros((len(feat),n_feat))
        for i in range(n_feat):
            feat_noise[:,i]=np.sum(feat_noise_all[:,v_used[i]],axis=1)
        m_orto=ortho_group.rvs(dim=n_feat)
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

        skf=StratifiedKFold(n_splits=4)
        ggg=-1
        xor_pre=[]
        for train, test in skf.split(feat_noise,xor):
            ggg=(ggg+1)
            supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
            mod=supp.fit(feat_noise[train],xor[train])
            xor_pre.append(supp.score(feat_noise[test],xor[test]))
        xor_vec[tt]=np.mean(np.array(xor_pre))
        orto_vec[tt]=m_orto

    perf_abs=abs(xor_vec-0.5)
    index_min=np.argmin(perf_abs)

    ############################################
    # Recompute everything with the best orto (some sort of validation?)
    feat_noise_all=(feat+np.random.normal(0,std,(np.shape(feat))))
    feat_noise=nan*np.zeros((len(feat),n_feat))
    for i in range(n_feat):
        feat_noise[:,i]=np.sum(feat_noise_all[:,v_used[i]],axis=1)
    m_orto=orto_vec[index_min]
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
    dic['feat_noise_all']=feat_noise_all
    dic['feat_noise']=feat_noise
    dic['feat_trans']=feat_trans
    dic['feat_binary']=feat_binary
    dic['feat_binary_orig']=feat_binary_orig
    dic['xor']=xor
    dic['lin']=lin
    return dic


def evaluate_abstraction(feat_decod,feat_binary):
    # Only for 2D. It is missing the diagonal terms                                                                                                                                                       
    perf=nan*np.zeros((2,2,2))
    for k in range(2): #Loop on "dichotomies"
      for kk in range(2): #Loop on ways to train this particular "dichotomy"
         ind_train=(feat_binary[:,k]==kk)
         ind_test=(~ind_train)
         k_task=abs(k-1)
         task=feat_binary[:,k_task]         
         supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf[0,k,kk]=supp.score(feat_decod[ind_train],task[ind_train])
         perf[1,k,kk]=supp.score(feat_decod[ind_test],task[ind_test])
    return np.mean(perf,axis=2)

# Evaluate Abstraction properly
def abstraction_2D(feat_decod,feat_binary,reg):
    exp_uq=np.unique(feat_binary,axis=0)
    feat_binary_exp=np.zeros(len(feat_binary))
    for t in range(len(feat_binary)):
        for tt in range((len(exp_uq))):
            gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
            if gg:
                feat_binary_exp[t]=tt
    #
    #dichotomies=np.array([[0,0,1,1],[0,1,0,1],[0,1,1,0]])
    #train_dich=np.array([[[0,2],[1,3],[0,3],[1,2]],[[0,1],[2,3],[0,3],[1,2]],[[0,1],[2,3],[0,2],[1,3]]])
    #test_dich=np.array([[[1,3],[0,2],[1,2],[0,3]],[[2,3],[0,1],[1,2],[0,3]],[[2,3],[0,1],[1,3],[0,2]]])
    dichotomies=np.array([[0,0,1,1],[0,1,0,1]])
    train_dich=np.array([[[0,2],[1,3]],[[0,1],[2,3]]])
    test_dich=np.array([[[1,3],[0,2]],[[2,3],[0,1]]])
    
    perf=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(4):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=reg,class_weight='balanced',solver='lbfgs')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
    return perf

# Evaluate Abstraction properly
def abstraction_3D(feat_decod,feat_binary):
    exp_uq=np.unique(feat_binary,axis=0)
    feat_binary_exp=np.zeros(len(feat_binary))
    for t in range(len(feat_binary)):
        for tt in range((len(exp_uq))):
            gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
            if gg:
                feat_binary_exp[t]=tt
    
    dichotomies=np.array([[0,0,0,0,1,1,1,1],
                          [0,0,1,1,0,0,1,1],
                          [0,1,0,1,0,1,0,1]])
    
    train_dich=np.array([[[0,2,3,4,6,7],[1,2,3,5,6,7],[0,1,3,4,5,7],[0,1,2,4,5,6]],
                         [[0,1,2,3,4,6],[0,2,4,5,6,7],[1,3,4,5,6,7],[0,1,2,3,5,7]],
                         [[2,3,4,5,6,7],[0,1,4,5,6,7],[0,1,2,3,6,7],[0,1,2,3,4,5]]])
                          
    test_dich=np.array([[[1,5],[0,4],[2,6],[3,7]],
                        [[5,7],[1,3],[0,2],[4,6]],
                        [[0,1],[2,3],[4,5],[6,7]]])
    
    perf=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1])|(feat_binary_exp==train_dich[k][kk][2])|(feat_binary_exp==train_dich[k][kk][3])|(feat_binary_exp==train_dich[k][kk][4])|(feat_binary_exp==train_dich[k][kk][5]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(8):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
    return perf

# def evaluate_abstraction_all(feat_decod,feat_binary):
#     exp_uq=np.unique(feat_binary,axis=0) # unique experimental conditions
#     #
#     feat_binary_exp=np.zeros(len(feat_binary))
#     for t in range(len(feat_binary)):
#         for tt in range((len(exp_uq))):
#             gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
#             if gg:
#                 feat_binary_exp[t]=tt
#     feat_binary_exp=np.array(feat_binary_exp,dtype=np.int16)
#     #
#     labels=np.zeros(len(exp_uq))
#     n_labels2=int(len(labels)/2)
#     labels[n_labels2:]=1
#     perm_uq_pre=np.array([i for i in multiset_permutations(labels)]) # all diferent colorings
#     perm_uq=perm_uq_pre[0:int(len(perm_uq_pre)/2)]
#     #
#     perf=nan*np.zeros((2,len(perm_uq),1000))
#     #perf=nan*np.zeros((2,1000,1000))
#     for i in range(len(perm_uq)): # Loop on all different colorings (3 if d=2, 35 if d=3)
#         print (perm_uq[i])
#         clases=np.array([perm_uq[i][k] for k in feat_binary_exp])
#         ind0_train_pre=np.array([k for k in combinations(np.where(perm_uq[i]==0)[0],n_labels2-1)])# Strategy of using all but one for training
#         ind1_train_pre=np.array([k for k in combinations(np.where(perm_uq[i]==1)[0],n_labels2-1)])
#         index_train_pre=np.array([np.concatenate((k,kk),axis=0) for k in ind0_train_pre for kk in ind1_train_pre])# all posible ways of training this particular coloring
#         for ii in range(len(index_train_pre)): # Loop on all possible ways of training and testing a particular coloring
#             print (ii,index_train_pre[ii])
#             index_train=np.array([])
#             for k in index_train_pre[ii]:
#                 index_train=np.hstack((index_train,np.where(np.sum(exp_uq[k]==feat_binary,axis=1)==len(feat_binary[0]))[0]))
#             index_test=np.delete(np.arange(len(feat_binary)),index_train)
#             index_train=np.array(index_train,dtype=np.int16)
#             index_test=np.array(index_test,dtype=np.int16)
#             supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
#             mod=supp.fit(feat_decod[index_train],clases[index_train])
#             perf[0,i,ii]=supp.score(feat_decod[index_train],clases[index_train])
#             perf[1,i,ii]=supp.score(feat_decod[index_test],clases[index_test])
#     dic={}
#     dic['performance']=np.nanmean(perf,axis=2)
#     dic['perm_uq']=perm_uq
#     return dic


# Creates the XOR task (N-dim Parity) based on the optimal class to split each feature
# It is more realistic but the classes are unbalanced becuase there is always a lot of "no contact" trials.
# It is easier than split median (for both the linear and the nonlinear classifier)
def create_xor_class(feat,clase,n_used,std,reg_vec,n_val,n_cv):
    n_feat=len(feat[0])*len(feat[0,0])
    feat_xor=(np.reshape(feat,(len(feat),n_feat)))
    feat_xor_noise=(feat_xor+np.random.normal(0,std,(np.shape(feat_xor))))
    
    feat_projec=nan*np.zeros(np.shape(feat_xor))
    feat_binary=nan*np.zeros(np.shape(feat_xor))
    info_feat=nan*np.zeros(n_feat)
    len_class=nan*np.zeros((2,n_feat))
    for j in range(len(feat_xor[0])):
      svm_cl=svm_sum_val(feat=feat_xor[:,(j):(j+1)],clase=clase,reg_vec=reg_vec,n_val=n_val,n_cv=n_cv)
      svm=svm_cl.svm()
      perf_test=svm['performance'][1]
      index_max=np.argmax(perf_test)
      info_feat[j]=svm['performance'][2][index_max]
      
      weights=svm['weights'][index_max]
      intercept=svm['intercept'][index_max]
      feat_projec[:,j]=(np.dot(feat_xor_noise[:,j],weights[0])+intercept*np.ones(len(feat_xor)))
      feat_binary[:,j][feat_projec[:,j]>0]=1
      feat_binary[:,j][feat_projec[:,j]<=0]=0
      len_class[0,j]=len(feat_binary[:,j][feat_binary[:,j]==1])
      len_class[1,j]=len(feat_binary[:,j][feat_binary[:,j]==0])
      
    feat_used=np.argsort(info_feat)[-n_used:]
    xor=np.sum(feat_binary[:,feat_used],axis=1)%2
    
    dic={}
    dic['feat_xor']=feat_xor[:,feat_used]
    dic['feat_xor_noise']=feat_xor_noise[:,feat_used]
    dic['feat_projec']=feat_projec[:,feat_used]
    dic['feat_binary']=feat_binary[:,feat_used]
    dic['feat_used']=feat_used
    dic['len_class']=len_class
    dic['xor']=xor
    return dic

def best_orto_rotation(feat,n,reg_vec,reg,n_cv,n_val,validation):
    n_feat=len(feat[0])
    mat_ort=nan*np.zeros((n,n_feat,n_feat))
    perf_xor=nan*np.zeros((n))
    for t in range(n):
        m_orto=ortho_group.rvs(dim=n_feat)
        mat_ort[t]=m_orto
        feat_rot=np.dot(feat,m_orto)
        feat_binary=nan*np.zeros(np.shape(feat))
        for p in range(n_feat):
            median_ct=np.median(feat_rot[:,p])
            feat_binary[:,p]=(feat_rot[:,p]>=median_ct)
        lin=feat_binary[:,0]
        xor=np.sum(feat_binary,axis=1)%2
        if validation:
            svm_cl=svm_sum_val(feat=feat,clase=xor,reg_vec=reg_vec,n_cv=n_cv,n_val=n_val)
            svm=svm_cl.svm(balanced=False)
            perf_test=svm['performance'][1]
            perf_val=svm['performance'][2]
            index_max=np.where(perf_test==np.nanmax(perf_test))[0][0]
            perf_xor[t]=perf_val[index_max]
        else:
            svm_cl=svm_reg(feat=feat,clase=xor,reg=reg,balanced=False,n_cv=n_cv)
            perf_xor[t]=svm_cl['performance'][1]

    perf_xor=abs(perf_xor-0.5)
    ind_ort=np.argmin(perf_xor)
    #print ('PERF ORTO ',perf_xor[ind_ort]+0.5)
    m_orto=mat_ort[ind_ort]
    return m_orto

def dimensionality_contacts(features,feat_binary,n_cv,n_color,n_rand):
    # Unique experimental conditions and each trial one integer that represents unique exp condition
    exp_uq=np.unique(feat_binary,axis=0) 
    feat_binary_exp=np.zeros(len(feat_binary))
    for t in range(len(feat_binary)):
        for tt in range((len(exp_uq))):
            gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
            if gg:
                feat_binary_exp[t]=tt
    feat_binary_exp=np.array(feat_binary_exp,dtype=np.int16)

    # We have n_exp unique experimental conditions and therefore n_part number of partitions
    n_exp=len(exp_uq)
    n_part=(2**n_exp)

    # Loop over all colorings and CV. CV is implemented such that the cv partition includes trials from all exp conditions
    dim_vec=nan*np.zeros((n_color,n_rand))
    iii=0
    while iii<n_color: # Loop on colorings
        try:
            color=np.array(np.random.normal(0,1,n_exp)>0,dtype=np.int16)
            clase_all=nan*np.zeros(len(feat_binary_exp))
            for oo in range(n_rand): # Loop on random CV partitions
                ind_train=np.array([])
                ind_test=np.array([])
                for ooo in range(n_exp): # On each CV partition both train and test are equally represented
                    ind_ooo=np.random.permutation(np.where(feat_binary_exp==ooo)[0])
                    ind_test=np.concatenate((ind_test,ind_ooo[0:int(len(ind_ooo)*1/n_cv)]))
                    ind_train=np.concatenate((ind_train,ind_ooo[int(len(ind_ooo)*1/n_cv):]))
                    clase_all[ind_ooo]=color[ooo]
                ind_train=np.array(ind_train,dtype=np.int16)
                ind_test=np.array(ind_test,dtype=np.int16)
            
            clf=LogisticRegression(C=1.0,class_weight='balanced')
            clf.fit(features[ind_train],clase_all[ind_train])
            dim_vec[iii,oo]=clf.score(features[ind_test],clase_all[ind_test])
            iii=(iii+1)
        except:
            None
    return np.nanmean(dim_vec)

#############################################################

def expand_weights(neurons,weights,bias):
    d1=len(weights) # number of neurons original
    d2=len(weights[0]) # number of neurons pre layer original
    #
    sig_w=torch.std(abs(weights))
    sig_b=torch.std(abs(bias))
    scale_std=1
    wei_def=torch.zeros((neurons,d2))
    bias_def=torch.zeros((neurons))
    for i in range(neurons):
        index_ch=torch.from_numpy(np.random.choice(np.arange(d1),1))
        wei_def[i]=torch.normal(weights[index_ch],scale_std*sig_w)
        bias_def[i]=torch.normal(bias[index_ch],scale_std*sig_b)
    return wei_def,bias_def
   

################################################################################
# Dimensionality Analysis

def dim_cv(n_split,data):
    num_trials=len(data)
    num_neu=len(data[0])
    expl_train=np.zeros((n_split,num_neu))
    expl_test=np.zeros((n_split,num_neu))
    expl_cumm_train=np.zeros((n_split,num_neu))
    expl_cumm_test=np.zeros((n_split,num_neu))
    dic={}

    cv=KFold(n_splits=n_split)
    g=0
    for train_index, test_index in cv.split(data):
        x_train=data[train_index]
        x_test=data[test_index]

        # Explained Variance Train set
        pca_clf=PCA(n_components=num_neu)
        pca=pca_clf.fit(x_train)
        expl_train[g]=pca.explained_variance_ratio_
        
        # Project test and train set onto eigenvectors from train set
        proj_data_train=np.transpose(np.dot(pca.components_,np.transpose(x_train)))
        proj_data_test=np.transpose(np.dot(pca.components_,np.transpose(x_test)))
        cov_proj_train=np.cov(proj_data_train,rowvar=False)
        cov_proj_test=np.cov(proj_data_test,rowvar=False)        
        
        # Variance explained on the test and train set
        var_train_cumm_pre=np.zeros(num_neu)
        count_train=0
        for j in range(num_neu):
            count_train=(count_train+cov_proj_train[j,j])
            var_train_cumm_pre[j]=count_train
        var_train_cumm=(var_train_cumm_pre/count_train)
            
        var_test_pre=np.zeros(num_neu)
        var_test_cumm_pre=np.zeros(num_neu)
        count_test=0
        for j in range(num_neu):
            count_test=(count_test+cov_proj_test[j,j])
            var_test_cumm_pre[j]=(count_test)
            var_test_pre[j]=cov_proj_test[j,j]
        var_test=(var_test_pre/count_test)
        var_test_cumm=(var_test_cumm_pre/count_test)            
        
        expl_test[g]=var_test
        expl_cumm_train[g]=var_train_cumm
        expl_cumm_test[g]=var_test_cumm
        g=g+1 
    dic['explained_var_train']=expl_train
    dic['explained_var_cumm_train']=expl_cumm_train
    dic['explained_var_test']=expl_test
    dic['explained_var_cumm_test']=expl_cumm_test
    return dic

def gini_index(x):
    '''
    Given a cummulative explained variance curve, it outputs the Gini dimensionality
    '''
    qua_vec=np.zeros(len(x)-1)
    for i in range(0,len(x)-1,1):
        qua_vec[i]=0.5*(x[i]+x[i+1])
        
    dim_pre=np.sum(qua_vec)/float(len(x))
    return 2*(1-dim_pre)

def rank(x):
    '''
    Given a cummulative explained variance curve, it outputs the rank
    '''
    eps=1e-4
    rest=(1.0-x)
    rank=len(rest[rest>eps])+1
    return rank

def rank_raw(data,eps,ratio):
    n_max=min([len(data),len(data[0])])
    pca_clf=PCA(n_components=n_max)
    pca=pca_clf.fit(data)
    if ratio:
        pc=pca.explained_variance_ratio_
    else:
        pc=pca.explained_variance_
    rank=len(pc[pc>eps])
    return rank

def participation_ratio_raw(data):
    n_max=min([len(data),len(data[0])])
    pca_clf=PCA(n_components=n_max)
    pca=pca_clf.fit(data)
    eigen=pca.explained_variance_
    num=np.sum(np.outer(eigen,eigen))
    den=np.sum(eigen**2)
    return num/den

def gini_raw(data):
    n_max=min([len(data),len(data[0])])
    pca_clf=PCA(n_components=n_max)
    pca=pca_clf.fit(data)
    pc=pca.explained_variance_ratio_
    cum=np.zeros(n_max)
    for i in range(0,n_max,1):
        cum[i]=np.sum(pc[0:(i+1)])
    
    # qua_vec=np.zeros(n_max-1)
    # for i in range(0,n_max-1,1):
    #     qua_vec[i]=0.5*(cum[i]+cum[i+1])
    # dim_pre=np.sum(qua_vec)/float(len(cum))

    dim_pre=np.sum(cum)/float(len(cum))
    return 2*(1-dim_pre)

#############################################################
# Genetic algorithm
def genetic_algorithm(model,sigma,neuron,n_reps,n_fit,n_trials,len_feat):
    dic={}
    out_evol=np.zeros(n_reps)
    population=sigma*torch.randn(len_feat)
    for i in range(n_reps):
        drift=sigma*torch.randn(n_trials,len_feat)
        X=(population*torch.ones((n_trials,len_feat))+drift)
        output=model(X)[:,neuron]
        out_evol[i]=np.mean(output.detach().numpy())
        ind_sort=torch.argsort(output)
        X_sort=X[ind_sort]
        population_pre=(population+torch.mean(X_sort[-n_fit:],dim=0))
        population=(population_pre/torch.norm(population_pre))
    dic['opt_stimulus']=population.detach().numpy()
    dic['out_evol']=out_evol
    return dic

def prob_quantity(quantity,lick_rate):
    num_steps=len(lick_rate[0])
    prob_quantity=nan*np.zeros((num_steps))
    for h in range(num_steps):
        ch_time_trials=np.sign(lick_rate[:,h])
        corr_st=(ch_time_trials+quantity)
        corr_st_v=abs(corr_st[(corr_st==2)|(corr_st==0)|(corr_st==-2)])
        n=len(corr_st_v)
        k=len(corr_st_v[corr_st_v==2])
        try:
            p=k/float(n)
            prob_quantity[h]=p
        except:
            None
    return prob_quantity

##########################################

def shatter_dim(data,condition):
    perf_thres=0.8
    n_cv=5
    #
    exp_cond_uq=np.unique(condition)
    words=["".join(seq) for seq in itertools.product("01", repeat=len(exp_cond_uq))]
    perf_vec=nan*np.zeros((len(words),n_cv))
    for i in range(len(words)): # Loop on all possible colors
        if i==0 or i==(len(words)-1):
            perf_vec[i]=np.ones(n_cv)
        else:    
            colors=np.zeros(len(condition))
            for ii in range(len(words[i])):
                colors[condition==exp_cond_uq[ii]]=int(words[i][ii])
            #
            skf=StratifiedKFold(n_splits=n_cv)
            g=0
            for train, test in skf.split(data,colors):
                supp=LinearSVC(dual=False,C=1e10,class_weight='balanced')
                supp.fit(data[train],colors[train])
                perf_vec[i,g]=supp.score(data[test],colors[test])
                g+=1

    perf_def=np.mean(perf_vec,axis=1)
    return np.mean(perf_def>=perf_thres)

# def confusion_matrix(data,condition):
#     n_cv=5
#     exp_cond_uq=np.unique(condition)
#     conf_mat_pre=nan*np.zeros((len(exp_cond_uq),len(exp_cond_uq),n_cv))
#     #
#     for i in range()
#     skf=StratifiedKFold(n_splits=n_cv)
#     g=0
#     for train, test in skf.split(data,colors):
#         supp=LinearSVC(dual=False,C=1e10,class_weight='balanced')
#         supp.fit(data[train],colors[train])
#         perf_vec[i,g]=supp.score(data[test],colors[test])
#         g+=1

#     perf_def=np.mean(perf_vec,axis=1)
#     return np.mean(perf_def>=perf_thres)

###############################################
# SVM validation

class svm_val:
    def __init__(self,feat,clase,reg_vec,n_val,n_cv):
        self.lenf=len(feat[0]) 
        self.lent=len(feat[0,0])
        self.feat=np.reshape(feat,(len(feat),self.lenf*self.lent))
        self.clase=clase
        self.reg_vec=reg_vec
        self.n_splits_val=n_val
        self.n_splits=n_cv
        
    def svm(self):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True) # 0 para train, 1 para test, 2 para validation                   
        perf=nan*np.zeros((3,self.n_splits_val,len(self.reg_vec),self.n_splits))
        weights=nan*np.zeros((self.n_splits_val,len(self.reg_vec),self.n_splits,self.lenf*self.lent))
        intercept=nan*np.zeros((self.n_splits_val,len(self.reg_vec),self.n_splits))
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            for hhh in range(len(self.reg_vec)):
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    #supp=LinearSVC(dual=False,C=1.0/self.reg_vec[hhh],class_weight='balanced')
                    supp=LogisticRegression(C=1.0/self.reg_vec[hhh],class_weight='balanced',solver='lbfgs')
                    mod=supp.fit(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[0,g_val,hhh,g]=supp.score(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[1,g_val,hhh,g]=supp.score(self.feat[train_index1][test_index],self.clase[train_index1][test_index])
                    perf[2,g_val,hhh,g]=supp.score(self.feat[val_index],self.clase[val_index])
                    weights[g_val,hhh,g]=supp.coef_[0]
                    intercept[g_val,hhh,g]=supp.intercept_[0]
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=(1,3)),'weights':np.nanmean(weights,axis=(0,2)),'intercept':np.nanmean(intercept,axis=(0,2))}
        return output

    def svm_nlin(self,model):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True) # 0 para train, 1 para test, 2 para validation                   
        perf=nan*np.zeros((3,self.n_splits_val,len(self.reg_vec),self.n_splits))
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            for hhh in range(len(self.reg_vec)):
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    supp=SVC(kernel=model,C=1.0/self.reg_vec[hhh],class_weight='balanced',gamma='auto')
                    mod=supp.fit(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[0,g_val,hhh,g]=supp.score(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[1,g_val,hhh,g]=supp.score(self.feat[train_index1][test_index],self.clase[train_index1][test_index])
                    perf[2,g_val,hhh,g]=supp.score(self.feat[val_index],self.clase[val_index])
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=(1,3))}
        return output

class svm_sum_val:
    def __init__(self,feat,clase,reg_vec,n_val,n_cv):
        self.lenf=len(feat[0])
        self.feat=feat
        self.clase=clase
        self.reg_vec=reg_vec
        self.n_splits_val=n_val
        self.n_splits=n_cv
        
    def svm(self,balanced):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True) # 0 para train, 1 para test, 2 para validation                   
        perf=nan*np.zeros((3,self.n_splits_val,len(self.reg_vec),self.n_splits))
        weights=nan*np.zeros((self.n_splits_val,len(self.reg_vec),self.n_splits,self.lenf))
        intercept=nan*np.zeros((self.n_splits_val,len(self.reg_vec),self.n_splits))
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            for hhh in range(len(self.reg_vec)):
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    #supp=LinearSVC(dual=False,C=1.0/self.reg_vec[hhh],class_weight='balanced')
                    #supp=LogisticRegression(C=1.0/self.reg_vec[hhh],class_weight='balanced',solver='lbfgs')
                    if balanced==True:
                        supp=LogisticRegression(C=1.0/self.reg_vec[hhh],class_weight='balanced',solver='lbfgs')
                    if balanced==False:
                        supp=LogisticRegression(C=1.0/self.reg_vec[hhh],solver='lbfgs')
                    mod=supp.fit(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[0,g_val,hhh,g]=supp.score(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[1,g_val,hhh,g]=supp.score(self.feat[train_index1][test_index],self.clase[train_index1][test_index])
                    perf[2,g_val,hhh,g]=supp.score(self.feat[val_index],self.clase[val_index])
                    weights[g_val,hhh,g]=supp.coef_[0]
                    intercept[g_val,hhh,g]=supp.intercept_[0]
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=(1,3)),'weights':np.nanmean(weights,axis=(0,2)),'intercept':np.nanmean(intercept,axis=(0,2))}
        return output

    def svm_nlin(self,model):
        dic_models={}
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True) # 0 para train, 1 para test, 2 para validation                   
        perf=nan*np.zeros((3,self.n_splits_val,len(self.reg_vec),self.n_splits))
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            for hhh in range(len(self.reg_vec)):
                cv=KFold(n_splits=self.n_splits,shuffle=True)
                g=0
                for train_index, test_index in cv.split(self.feat[train_index1]):
                    supp=SVC(kernel=model,C=1.0/self.reg_vec[hhh],class_weight='balanced',gamma='auto')
                    mod=supp.fit(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[0,g_val,hhh,g]=supp.score(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                    perf[1,g_val,hhh,g]=supp.score(self.feat[train_index1][test_index],self.clase[train_index1][test_index])
                    perf[2,g_val,hhh,g]=supp.score(self.feat[val_index],self.clase[val_index])
                    g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=(1,3))}
        return output

#######################################
# Load Models
def load_models(abs_path_save,loss_type,arx,mod_arch,type_trials,ablation,reg_vec,len_feat,num_neu):
    perf_reg=nan*np.zeros((3,len(reg_vec),20))
    for iii in range(len(reg_vec)):
        try:
            apn=open(abs_path_save+'performance_nonlinear_%s_%s_%s_%i%s%s.pkl'%(loss_type,arx,mod_arch,iii,type_trials,ablation),'rb')
            perf_init=pkl.load(apn)['performance']
            apn.close()
            perf_init_pre=(1-perf_init[:,0]/perf_init[:,1])
            perf_pre=np.nanmean(perf_init_pre,axis=(1,2))
            perf_reg[:,iii]=perf_pre
        except:
            None

    index_max=np.where(perf_reg[1]==np.nanmax(perf_reg[1]))        
    #print (index_max)
    index_max_reg=index_max[0][0]
    index_max_lr=index_max[1][0]
    num_neu=len(perf_init_pre[1])
    
    arx_models=torch.load(abs_path_save+'models_nonlinear_%s_%s_%s_%i%s%s.pt'%(loss_type,arx,mod_arch,index_max_reg,type_trials,ablation),map_location='cpu')
    
    if mod_arch=='FC_trivial':
        model=nn_pytorch_encoders.FC_trivial(len_feat=len_feat,len_target=num_neu)
    if mod_arch=='FC_one_deep_hundred_wide':
        model=nn_pytorch_encoders.FC_one_deep_hundred_wide(len_feat=len_feat,len_target=num_neu)
    if mod_arch=='FC_two_deep_hundred_wide':
        model=nn_pytorch_encoders.FC_two_deep_hundred_wide(len_feat=len_feat,len_target=num_neu)
    if mod_arch=='FC_three_deep_hundred_wide':
        model=nn_pytorch_encoders.FC_three_deep_hundred_wide(len_feat=len_feat,len_target=num_neu)
                
    apn=open(abs_path_save+'index_nonlinear_%s_%s_%s_%i%s%s.pkl'%(loss_type,arx,mod_arch,index_max_reg,type_trials,ablation),'rb')
    index_cv=pkl.load(apn)['index']
    dic={}
    dic['arx_models']=arx_models
    dic['model']=model
    dic['index_cv']=index_cv
    dic['index_max_lr']=index_max_lr
    return dic
           

#########################################################
# Casual Neural Nets
class neural_net:
    def __init__(self,feat,clase,reg_vec,lr_vec,n_val,n_cv):
        self.lenf=len(feat[0])
        self.feat=feat
        self.clase=clase
        self.reg_vec=reg_vec
        self.lr_vec=lr_vec
        self.n_splits_val=n_val
        self.n_splits=n_cv
        self.batch_size=64
        
    def nn(self,model):
        cv_val=KFold(n_splits=self.n_splits_val,shuffle=True) # 0 para train, 1 para test, 2 para validation                   
        perf=nan*np.zeros((3,self.n_splits_val,len(self.reg_vec),len(self.lr_vec),self.n_splits))
        g_val=0
        for train_index1, val_index in cv_val.split(self.feat):
            print ('Validation ',g_val)
            for h in range(len(self.reg_vec)):
                print ('  REG ',self.reg_vec[h])
                for hh in range(len(self.lr_vec)):
                    #print ('  LR ',self.lr_vec[hh])
                    cv=KFold(n_splits=self.n_splits,shuffle=True)
                    g=0
                    for train_index, test_index in cv.split(self.feat[train_index1]):
                        supp=nn_feedforward(type_class=model,reg=self.reg_vec[h],lr=self.lr_vec[hh],len_feat=self.lenf)
                        mod=supp.fit(self.feat[train_index1][train_index],self.clase[train_index1][train_index],batch_size=self.batch_size)
                        perf[0,g_val,h,hh,g]=supp.score(self.feat[train_index1][train_index],self.clase[train_index1][train_index])
                        perf[1,g_val,h,hh,g]=supp.score(self.feat[train_index1][test_index],self.clase[train_index1][test_index])
                        perf[2,g_val,h,hh,g]=supp.score(self.feat[val_index],self.clase[val_index])
                        #print ('    ',perf[:,g_val,h,hh,g])
                        g=g+1
            g_val=g_val+1
        output={'performance':np.nanmean(perf,axis=(1,4))}
        return output

class nn_feedforward():
    def __init__(self,type_class,reg,lr,len_feat):
        self.type_class=type_class
        self.regularization=reg
        self.learning_rate=lr
        self.len_feat=len_feat

        # Fully Connected
        if type_class=='FC_trivial':
            self.model=FC_trivial(n_feat=self.len_feat)
        if type_class=='FC_one_deep_hundred_wide':
            self.model=FC_one_deep_hundred_wide(n_feat=self.len_feat)
        if type_class=='FC_two_deep_hundred_wide':
            self.model=FC_two_deep_hundred_wide(n_feat=self.len_feat)
        if type_class=='FC_three_deep_hundred_wide':
            self.model=FC_three_deep_hundred_wide(n_feat=self.len_feat)
        if type_class=='FC_four_deep_hundred_wide':
            self.model=FC_four_deep_hundred_wide(n_feat=self.len_feat)
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self,feat,clase,batch_size):
        self.model.train()
        clase_np=np.array(clase,dtype=np.int64)
        clase_np[clase_np==-1]=0
        feat_np=np.array(feat,dtype=np.float32)
        clase_torch=Variable(torch.from_numpy(clase_np),requires_grad=False)#.cuda()                                                                                                                     
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        # Deal with class imbalance and equal correct and incorrect
        class_count=np.array([len(np.where(clase_np==i)[0]) for i in np.unique(clase_np)])
        weight=1.0/class_count
        samples_weight=torch.from_numpy(np.array([weight[t] for t in clase_np])).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight,int(np.min(class_count)),replacement=False)
        train_loader=DataLoader(torch.utils.data.TensorDataset(feat_torch,clase_torch),batch_size=batch_size,shuffle=False,sampler=sampler)
                                                                                                                                                                                                          
        t_total=100
        for t in range(t_total):
            # if t==0:
            #     print ('    ini ',self.loss(self.model(feat_torch),clase_torch).item())
            for batch_idx, (data, targets) in enumerate(train_loader):
                self.loss(self.model(data),targets).item()
                self.optimizer.zero_grad()
                loss=torch.mean(self.loss(self.model(data),targets))
                loss.backward()
                self.optimizer.step()
            # if t==(t_total-1):
            #     print ('    fin ',self.loss(self.model(feat_torch),clase_torch).item())
        return self.model.state_dict()

    def score(self,feat,clase):                                              
        self.model.eval()
        clase_np=np.array(clase,dtype=np.int64)
        clase_np[clase_np==-1]=0
        feat_np=np.array(feat,dtype=np.float32)
        clase_torch=Variable(torch.from_numpy(clase_np),requires_grad=False)#.cuda()                                                                                                                      
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        test_loader=DataLoader(torch.utils.data.TensorDataset(feat_torch,clase_torch),batch_size=len(feat_np),shuffle=False)
        
        for batch_idx, (data,targets) in enumerate(test_loader):
            y_pred=np.argmax(self.model(data).detach().numpy(),axis=1)
            target_np=targets.detach().numpy()
            error=np.mean(abs(y_pred-target_np))
        return 1.0-error

# Network Models
class FC_trivial(torch.nn.Module):
    def __init__(self,n_feat):
        super(FC_trivial,self).__init__()
        self.linear1=torch.nn.Linear(n_feat,2)

    def forward(self,x):
        x = self.linear1(x)
        return x

class FC_one_deep_hundred_wide(torch.nn.Module):
    def __init__(self,n_feat):
        super(FC_one_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat,100)
        self.linear2=torch.nn.Linear(100,2)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class FC_two_deep_hundred_wide(torch.nn.Module):
    def __init__(self,n_feat):
        super(FC_two_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat,100)
        self.linear2=torch.nn.Linear(100,100)
        self.linear3=torch.nn.Linear(100,2)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class FC_three_deep_hundred_wide(torch.nn.Module):
    def __init__(self,n_feat):
        super(FC_three_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat,100)
        self.linear2=torch.nn.Linear(100,100)
        self.linear3=torch.nn.Linear(100,100)
        self.linear4=torch.nn.Linear(100,2)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class FC_four_deep_hundred_wide(torch.nn.Module):
    def __init__(self,n_feat):
        super(FC_four_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat,100)
        self.linear2=torch.nn.Linear(100,100)
        self.linear3=torch.nn.Linear(100,100)
        self.linear4=torch.nn.Linear(100,100)
        self.linear5=torch.nn.Linear(100,2)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x
