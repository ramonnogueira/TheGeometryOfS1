import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy
#import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
nan=float('nan')

dtype = torch.FloatTensor

class nn_feedforward():
    def __init__(self,type_class,reg,lr,len_feat,len_time):
        self.type_class=type_class
        self.regularization=reg
        self.learning_rate=lr
        self.len_feat=len_feat
        self.len_time=len_time

        # Fully Connected
        if type_class=='FC_trivial':
            self.model=FC_trivial(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='FC_one_deep_two_wide':
            self.model=FC_one_deep_twenty_wide(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='FC_one_deep_twenty_wide':
            self.model=FC_one_deep_twenty_wide(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='FC_one_deep_hundred_wide':
            self.model=FC_one_deep_hundred_wide(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='FC_two_deep_twenty_wide':
            self.model=FC_two_deep_twenty_wide(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='FC_two_deep_hundred_wide':
            self.model=FC_two_deep_hundred_wide(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='FC_three_deep_twenty_wide':
            self.model=FC_three_deep_twenty_wide(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='FC_three_deep_hundred_wide':
            self.model=FC_three_deep_hundred_wide(n_feat=self.len_feat,n_time=self.len_time)
        # Convolutional
        if type_class=='conv_net_one_layer':
            self.model=conv_net_one_layer(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='conv_net_two_layer':
            self.model=conv_net_two_layer(n_feat=self.len_feat,n_time=self.len_time)
        if type_class=='conv_net_three_layer':
            self.model=conv_net_three_layer(n_feat=self.len_feat,n_time=self.len_time)

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self,feat,clase,batch_size,reward):
        self.model.train()
        clase_np=np.array(clase,dtype=np.int64)
        clase_np[clase_np==-1]=0
        reward[reward==-1]=0
        reward_np=np.array(reward,dtype=np.float32)
        feat_np=np.array(feat,dtype=np.float32)
        clase_torch=Variable(torch.from_numpy(clase_np),requires_grad=False)#.cuda()                                                                                                                     
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        reward_torch=Variable(torch.from_numpy(reward_np),requires_grad=False)#.cuda()
        # Deal with class imbalance and equal correct and incorrect
        values=nan*np.zeros(len(clase_np))
        values[(reward==0)&(clase_np==0)]=0
        values[(reward==0)&(clase_np==1)]=1
        values[(reward==1)&(clase_np==0)]=2
        values[(reward==1)&(clase_np==1)]=3
        values=np.array(values,dtype=np.int16)
        min_class=np.nanmin([len(np.where(values==0)[0]),len(np.where(values==1)[0]),len(np.where(values==2)[0]),len(np.where(values==3)[0])])
                                                                                                                                                                                                          
        t_total=100
        for t in range(t_total):
            index_t=np.array([])
            for tt in range(4):
                index_t=np.concatenate((index_t,np.random.choice(np.where(values==tt)[0],size=min_class,replace=False)))
            index_t=np.sort(index_t)
            index_t=np.array(index_t,dtype=np.int16)
            n_it=int(len(index_t)/batch_size)
            rest=int(len(index_t)%batch_size)
            #
            #if t==0:
            #    print ('  ini ',self.loss(self.model(feat_torch),clase_torch).item())
            for ttt in range(n_it):
                self.optimizer.zero_grad()
                loss=torch.mean(self.loss(self.model(feat_torch[index_t[batch_size*ttt:batch_size*(ttt+1)]]),clase_torch[index_t[batch_size*ttt:batch_size*(ttt+1)]]))
                loss.backward()
                self.optimizer.step()
            if rest!=0:
                self.optimizer.zero_grad()
                loss=torch.mean(self.loss(self.model(feat_torch[index_t[batch_size*n_it:]]),clase_torch[index_t[batch_size*n_it:]]))
                loss.backward()
                self.optimizer.step()
            #if t==(t_total-1):
            #    print ('  fin ',self.loss(self.model(feat_torch),clase_torch).item())
        return self.model.state_dict()

    def fit_no_equal(self,feat,clase,batch_size):
        self.model.train()
        clase_np=np.array(clase,dtype=np.int64)
        clase_np[clase_np==-1]=0
        feat_np=np.array(feat,dtype=np.float32)
        clase_torch=Variable(torch.from_numpy(clase_np),requires_grad=False)#.cuda()                                                                                                                     
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        test_loader=DataLoader(torch.utils.data.TensorDataset(feat_torch,clase_torch),batch_size=batch_size,shuffle=True)
                                                                                                                                                                                                         
        t_total=100
        for t in range(t_total):
            #if t==0:
            #    print ('  ini ',self.loss(self.model(feat_torch),clase_torch).item())
            for batch_idx, (data,targets) in enumerate(test_loader):
                self.optimizer.zero_grad()
                loss=torch.mean(self.loss(self.model(data),targets))
                loss.backward()
                self.optimizer.step()
            #if t==(t_total-1):
            #    print ('  fin ',self.loss(self.model(feat_torch),clase_torch).item())
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

    def fit_corr_incorr(self,feat,clase,batch_size,reward,corr):
        self.model.train()
        clase_np=np.array(clase,dtype=np.int64)
        clase_np[clase_np==-1]=0
        reward_np=np.array(reward,dtype=np.int64)
        reward_np[reward_np==-1]=0
        feat_np=np.array(feat,dtype=np.float32)
        clase_torch=Variable(torch.from_numpy(clase_np),requires_grad=False)#.cuda()                                                                                                                     
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        # Uses corrects or incorrects only
        values=nan*np.zeros(len(clase_np))
        if corr==True:
            values[(reward_np==1)&(clase_np==0)]=0
            values[(reward_np==1)&(clase_np==1)]=1
            values[(reward_np==0)&(clase_np==0)]=2
            values[(reward_np==0)&(clase_np==1)]=3
        if corr==False:
            values[(reward_np==0)&(clase_np==0)]=0
            values[(reward_np==0)&(clase_np==1)]=1
            values[(reward_np==1)&(clase_np==0)]=2
            values[(reward_np==1)&(clase_np==1)]=3
        values=np.array(values,dtype=np.int16)
        min_class=np.nanmin([len(np.where(values==0)[0]),len(np.where(values==1)[0]),len(np.where(values==2)[0]),len(np.where(values==3)[0])])
        
        t_total=100
        for t in range(t_total):
            index_t=np.array([])
            for tt in range(2):
                index_t=np.concatenate((index_t,np.random.choice(np.where(values==tt)[0],size=min_class,replace=False)))
            index_t=np.sort(index_t)
            index_t=np.array(index_t,dtype=np.int16)
            n_it=int(len(index_t)/batch_size)
            rest=int(len(index_t)%batch_size)
            #
            if t==0:
                print ('  ini ',self.loss(self.model(feat_torch),clase_torch).item())
            for ttt in range(n_it):
                self.optimizer.zero_grad()
                loss=torch.mean(self.loss(self.model(feat_torch[index_t[batch_size*ttt:batch_size*(ttt+1)]]),clase_torch[index_t[batch_size*ttt:batch_size*(ttt+1)]]))
                loss.backward()
                self.optimizer.step()
            if rest!=0:
                self.optimizer.zero_grad()
                loss=torch.mean(self.loss(self.model(feat_torch[index_t[batch_size*n_it:]]),clase_torch[index_t[batch_size*n_it:]]))
                loss.backward()
                self.optimizer.step()
            if t==(t_total-1):
                print ('  fin ',self.loss(self.model(feat_torch),clase_torch).item())
        return self.model.state_dict()

    def score_corr_incorr(self,feat,clase,reward,corr):                                              
        self.model.eval()
        clase_np=np.array(clase,dtype=np.int64)
        clase_np[clase_np==-1]=0
        reward_np=np.array(reward,dtype=np.int64)
        reward_np[reward_np==-1]=0
        feat_np=np.array(feat,dtype=np.float32)
        clase_torch=Variable(torch.from_numpy(clase_np),requires_grad=False)#.cuda()                                                                                                                      
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        # Uses corrects or incorrects only
        if corr==True:
            n_batch=len(reward_np[reward_np==1])
            samples_weight=torch.from_numpy(reward_np).double()
        if corr==False:
            n_batch=len(reward_np[reward_np==0])
            samples_weight=torch.from_numpy(np.ones(len(reward_np))-reward_np).double()
        sampler=torch.utils.data.sampler.WeightedRandomSampler(samples_weight,n_batch,replacement=False)
        test_loader=DataLoader(torch.utils.data.TensorDataset(feat_torch,clase_torch),batch_size=n_batch,shuffle=False,sampler=sampler)

        for batch_idx, (data,targets) in enumerate(test_loader):
            y_pred=np.argmax(self.model(data).detach().numpy(),axis=1)
            target_np=targets.detach().numpy()
            error=np.mean(abs(y_pred-target_np))
        return 1.0-error

########################################################
# Network Models

class FC_trivial(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_trivial,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,2)

    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0])) # This [:,0] is to make it compatible with convolutional architectures
        x = self.linear1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FC_one_deep_two_wide(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_one_deep_twenty_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,2)
        self.linear2=torch.nn.Linear(2,2)
        
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0]))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class FC_one_deep_twenty_wide(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_one_deep_twenty_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,20)
        self.linear2=torch.nn.Linear(20,2)
        
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0]))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FC_one_deep_hundred_wide(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_one_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,100)
        self.linear2=torch.nn.Linear(100,2)
        
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0]))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FC_two_deep_twenty_wide(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_two_deep_twenty_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,20)
        self.linear2=torch.nn.Linear(20,20)
        self.linear3=torch.nn.Linear(20,2)
        
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0]))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FC_two_deep_hundred_wide(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_two_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,100)
        self.linear2=torch.nn.Linear(100,100)
        self.linear3=torch.nn.Linear(100,2)
        
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0]))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FC_three_deep_twenty_wide(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_three_deep_twenty_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,20)
        self.linear2=torch.nn.Linear(20,20)
        self.linear3=torch.nn.Linear(20,20)
        self.linear4=torch.nn.Linear(20,2)
        
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0]))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FC_three_deep_hundred_wide(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(FC_three_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(n_feat*n_time,100)
        self.linear2=torch.nn.Linear(100,100)
        self.linear3=torch.nn.Linear(100,100)
        self.linear4=torch.nn.Linear(100,2)
        
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x[:,0]))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class conv_net_one_layer(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(conv_net_one_layer,self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(n_feat,int(n_time/4)),stride=1)
        self.pool1=torch.nn.MaxPool2d(kernel_size=(1,int(n_time/4)),stride=1)
        self.linear1=torch.nn.Linear(2*12,20)
        self.linear2=torch.nn.Linear(20,2)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class conv_net_two_layer(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(conv_net_two_layer,self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=5,kernel_size=(n_feat,int(n_time/4)),stride=1)
        self.pool1=torch.nn.MaxPool2d(kernel_size=(1,int(n_time/4)),stride=1)
        self.conv2=torch.nn.Conv2d(in_channels=5,out_channels=10,kernel_size=(1,int(n_time/6)),stride=1)
        self.pool2=torch.nn.MaxPool2d(kernel_size=(1,int(n_time/6)),stride=1)
        self.linear1=torch.nn.Linear(10*8,100)
        self.linear2=torch.nn.Linear(100,2)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class conv_net_three_layer(torch.nn.Module):
    def __init__(self,n_feat,n_time):
        super(conv_net_three_layer,self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=5,kernel_size=(n_feat,int(n_time/4)),stride=1)
        self.pool1=torch.nn.MaxPool2d(kernel_size=(1,int(n_time/4)),stride=1)
        self.conv2=torch.nn.Conv2d(in_channels=5,out_channels=10,kernel_size=(1,int(n_time/6)),stride=1)
        self.pool2=torch.nn.MaxPool2d(kernel_size=(1,int(n_time/6)),stride=1)
        self.conv3=torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(1,int(n_time/8)),stride=1)
        self.pool3=torch.nn.MaxPool2d(kernel_size=(1,int(n_time/8)),stride=1)
        self.linear1=torch.nn.Linear(120,100)
        self.linear2=torch.nn.Linear(100,2)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


