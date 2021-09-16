import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy
from torch.utils.data import DataLoader
nan=float('nan')

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

class nn_encoding():
    def __init__(self,type_class,reg,lr,loss_type,len_feat,len_target):
        self.regularization=reg
        self.learning_rate=lr
        self.loss_type=loss_type
        self.len_feat=len_feat
        self.len_target=len_target

        if type_class=='FC_trivial':
            self.model=FC_trivial(len_feat=self.len_feat,len_target=self.len_target)#.cuda()
        if type_class=='FC_one_deep_twenty_wide':
            self.model=FC_one_deep_twenty_wide(len_feat=self.len_feat,len_target=self.len_target)#.cuda()
        if type_class=='FC_one_deep_hundred_wide':
            self.model=FC_one_deep_hundred_wide(len_feat=self.len_feat,len_target=self.len_target)#.cuda()
        if type_class=='FC_two_deep_twenty_wide':
            self.model=FC_two_deep_twenty_wide(len_feat=self.len_feat,len_target=self.len_target)#.cuda()
        if type_class=='FC_two_deep_hundred_wide':
            self.model=FC_two_deep_hundred_wide(len_feat=self.len_feat,len_target=self.len_target)#.cuda()
        if type_class=='FC_three_deep_twenty_wide':
            self.model=FC_three_deep_twenty_wide(len_feat=self.len_feat,len_target=self.len_target)#.cuda()
        if type_class=='FC_three_deep_hundred_wide':
            self.model=FC_three_deep_hundred_wide(len_feat=self.len_feat,len_target=self.len_target)#.cuda()

        if self.loss_type=='gaussian':
            self.loss = torch.nn.MSELoss()
            self.loss_score = torch.nn.MSELoss(reduction='none')
        if self.loss_type=='poisson':
            self.loss = torch.nn.PoissonNLLLoss(log_input=True,full=True)
            self.loss_score = torch.nn.PoissonNLLLoss(log_input=True,full=True,reduction='none')
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self,feat,target,batch_size):
        self.model.train()
        feat_np=np.array(feat,dtype=np.float32)
        target_np=np.array(target,dtype=np.float32)
        target_torch=Variable(torch.from_numpy(target_np),requires_grad=False)#.cuda()
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        train_loader=DataLoader(torch.utils.data.TensorDataset(feat_torch,target_torch),batch_size=batch_size,shuffle=True)
        
        t_total=100
        for t in range(t_total):
            #print (t)
            #print (self.loss(self.model(feat_torch),target_torch).item())
            if t==0:
                print ('    ini ',self.loss(self.model(feat_torch),target_torch).item())
            for batch_idx, (data, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                loss=self.loss(self.model(data),targets)
                loss.backward()
                self.optimizer.step()
            if t==(t_total-1):
                print ('    fin ',self.loss(self.model(feat_torch),target_torch).item())
        return self.model.state_dict()

    def score(self,x,y):
        self.model.eval()
        x_np=np.array(x,dtype=np.float32)
        y_np=np.array(y,dtype=np.float32)
        x_torch=Variable(torch.from_numpy(x_np),requires_grad=False)#.cuda()
        y_torch=Variable(torch.from_numpy(y_np),requires_grad=False)#.cuda()
        test_loader=DataLoader(torch.utils.data.TensorDataset(x_torch,y_torch),batch_size=len(x_torch),shuffle=False)
        for batch_idx, (data,targets) in enumerate(test_loader):
            if self.loss_type=='gaussian':
                y_mean_torch=torch.mean(y_torch,dim=0)*torch.ones(y_torch.size())  
            if self.loss_type=='poisson':
                y_mean_torch=torch.log(torch.mean(y_torch,dim=0)*torch.ones(y_torch.size()))  
            loss=torch.mean(self.loss_score(self.model(data),targets),dim=0)
            loss_null=torch.mean(self.loss_score(y_mean_torch,targets),dim=0)
        return loss.detach().numpy(),loss_null.detach().numpy()

    def fit_corr_incorr(self,feat,target,batch_size,reward,corr):
        self.model.train()
        feat_np=np.array(feat,dtype=np.float32)
        target_np=np.array(target,dtype=np.float32)
        reward_np=np.array(reward,dtype=np.int64)
        target_torch=Variable(torch.from_numpy(target_np),requires_grad=False)#.cuda()
        feat_torch=Variable(torch.from_numpy(feat_np),requires_grad=False)#.cuda()
        #
        index_cor=np.where(reward_np==1)[0]
        index_inc=np.where(reward_np==-1)[0]
        min_class=np.nanmin([len(index_cor),len(index_inc)])
        if corr==True:
            index_use=index_cor.copy()
        if corr==False:
            index_use=index_inc.copy()
        
        t_total=100
        for t in range(t_total):
            #print (t)
            index_t=np.random.choice(index_use,size=min_class,replace=False)
            index_t=np.array(index_t,dtype=np.int16)
            n_it=int(len(index_t)/batch_size)
            rest=int(len(index_t)%batch_size)
            print (len(index_t))
            print (n_it)
            print (rest)
            print (reward_np[index_t])
            if t==0:
                print ('    ini ',self.loss(self.model(feat_torch),target_torch).item())
            for ttt in range(n_it):
                self.optimizer.zero_grad()
                loss=self.loss(self.model(feat_torch[index_t[batch_size*ttt:batch_size*(ttt+1)]]),target_torch[index_t[batch_size*ttt:batch_size*(ttt+1)]])
                loss.backward()
                self.optimizer.step()
            if rest!=0:
                self.optimizer.zero_grad()
                loss=self.loss(self.model(feat_torch[index_t[batch_size*n_it:]]),target_torch[index_t[batch_size*n_it:]])
                loss.backward()
                self.optimizer.step()
            if t==(t_total-1):
                print ('    fin ',self.loss(self.model(feat_torch),target_torch).item())
        return self.model.state_dict()

    def score_corr_incorr(self,x,y,reward,corr):
        self.model.eval()
        x_np=np.array(x,dtype=np.float32)
        y_np=np.array(y,dtype=np.float32)
        x_torch=Variable(torch.from_numpy(x_np),requires_grad=False)#.cuda()
        y_torch=Variable(torch.from_numpy(y_np),requires_grad=False)#.cuda()
        reward_np=np.array(reward,dtype=np.int64)
        
        index_cor=np.where(reward_np==1)[0]
        index_inc=np.where(reward_np==-1)[0]
        min_class=np.nanmin([len(index_cor),len(index_inc)])
        if corr==True:
            index_use=index_cor.copy()
        if corr==False:
            index_use=index_inc.copy()
            
        if self.loss_type=='gaussian':
            y_mean_torch=torch.mean(y_torch[index_use],dim=0)*torch.ones(y_torch[index_use].size())  
        if self.loss_type=='poisson':
            y_mean_torch=torch.log(torch.mean(y_torch[index_use],dim=0)*torch.ones(y_torch[index_use].size()))
            
        loss=torch.mean(self.loss_score(self.model(x_torch[index_use]),y_torch[index_use]),dim=0)
        loss_null=torch.mean(self.loss_score(y_mean_torch,y_torch[index_use]),dim=0)
        
        return loss.detach().numpy(),loss_null.detach().numpy()

# Network Architectures Encoder
class FC_trivial(torch.nn.Module):
    def __init__(self,len_feat,len_target):
        super(FC_trivial,self).__init__()
        self.linear1=torch.nn.Linear(len_feat,len_target)

    def forward(self,x):
        x = self.linear1(x)
        return x
    
class FC_one_deep_twenty_wide(torch.nn.Module):
    def __init__(self,len_feat,len_target):
        super(FC_one_deep_twenty_wide,self).__init__()
        self.linear1=torch.nn.Linear(len_feat,20)
        self.linear2=torch.nn.Linear(20,len_target)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class FC_one_deep_hundred_wide(torch.nn.Module):
    def __init__(self,len_feat,len_target):
        super(FC_one_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(len_feat,100)
        self.linear2=torch.nn.Linear(100,len_target)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class FC_two_deep_twenty_wide(torch.nn.Module):
    def __init__(self,len_feat,len_target):
        super(FC_two_deep_twenty_wide,self).__init__()
        self.linear1=torch.nn.Linear(len_feat,20)
        self.linear2=torch.nn.Linear(20,20)
        self.linear3=torch.nn.Linear(20,len_target)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class FC_two_deep_hundred_wide(torch.nn.Module):
    def __init__(self,len_feat,len_target):
        super(FC_two_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(len_feat,100)
        self.linear2=torch.nn.Linear(100,100)
        self.linear3=torch.nn.Linear(100,len_target)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class FC_three_deep_twenty_wide(torch.nn.Module):
    def __init__(self,len_feat,len_target):
        super(FC_three_deep_twenty_wide,self).__init__()
        self.linear1=torch.nn.Linear(len_feat,20)
        self.linear2=torch.nn.Linear(20,20)
        self.linear3=torch.nn.Linear(20,20)
        self.linear4=torch.nn.Linear(20,len_target)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class FC_three_deep_hundred_wide(torch.nn.Module):
    def __init__(self,len_feat,len_target):
        super(FC_three_deep_hundred_wide,self).__init__()
        self.linear1=torch.nn.Linear(len_feat,100)
        self.linear2=torch.nn.Linear(100,100)
        self.linear3=torch.nn.Linear(100,100)
        self.linear4=torch.nn.Linear(100,len_target)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
