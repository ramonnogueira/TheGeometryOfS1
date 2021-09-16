import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy
#import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
nan=float('nan')

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# Input shape has to be: Batch x Steps x Input dim
# Target shape has to be: Batch x Steps (x Target dim)
# The output of the model concatenates all time steps from all trials

class nn_recurrent():
    def __init__(self,reg,lr,input_size,output_size,hidden_dim,n_layers,t_dec):
        self.regularization=reg
        self.learning_rate=lr
        self.loss=torch.nn.CrossEntropyLoss()
        self.model=recurrent_noisy(input_size,output_size,hidden_dim,n_layers,t_dec)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self,input_seq,target_seq,batch_size,sigma_noise):
        thres_fit=1e-3
        self.model.train()
        input_seq_np=np.array(input_seq,dtype=np.float32)
        target_seq_np=np.array(target_seq,dtype=np.int16)
        input_seq_torch=Variable(torch.from_numpy(input_seq_np),requires_grad=False)
        target_seq_torch=Variable(torch.from_numpy(target_seq_np),requires_grad=False)
        train_loader=DataLoader(torch.utils.data.TensorDataset(input_seq_torch,target_seq_torch),batch_size=batch_size,shuffle=True)
        
        t_total=100
        for t in range(t_total):
            #if t==0:
            #    output, hidden, net_units, read_out_units = self.model(input_seq_torch,sigma_noise)
            #    print ('  ini ',self.loss(output,target_seq_torch.view(-1).long()).item())
            output, hidden, net_units, read_out_units = self.model(input_seq_torch,sigma_noise)
            #print (t,self.loss(output,target_seq_torch.view(-1).long()).item())
            if (self.loss(output,target_seq_torch.view(-1).long()).item())<thres_fit:
                break
            for batch_idx, (data, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output, hidden, net_units, read_out_units = self.model(data,sigma_noise)
                loss=self.loss(output,targets.view(-1).long())
                loss.backward()
                self.optimizer.step()
            #if t==(t_total-1):
            #    output, hidden, net_units, read_out_units = self.model(input_seq_torch,sigma_noise)
            #    print ('  fin ',self.loss(output,target_seq_torch.view(-1).long()).item())
        return self.model.state_dict()

    def score(self,input_seq,target_seq,sigma_noise):                                              
        self.model.eval()
        input_seq_np=np.array(input_seq,dtype=np.float32)
        target_seq_np=np.array(target_seq,dtype=np.int16)
        input_seq_torch=Variable(torch.from_numpy(input_seq_np),requires_grad=False)
        target_seq_torch=Variable(torch.from_numpy(target_seq_np),requires_grad=False)
        test_loader=DataLoader(torch.utils.data.TensorDataset(input_seq_torch,target_seq_torch),batch_size=len(target_seq),shuffle=False)
        
        for batch_idx, (data,targets) in enumerate(test_loader):
            output, hidden, net_units, read_out_units = self.model(data,sigma_noise)
            y_pred=np.argmax(output.detach().numpy(),axis=1)
            target_np=targets.detach().numpy()
            error=np.mean(abs(y_pred-target_np))
        return 1.0-error

    
class recurrent_noisy(torch.nn.Module): # We always send the input with size batch x time steps x input dim
    def __init__(self, input_size, output_size, hidden_dim, n_layers, t_dec):
        super(recurrent_noisy, self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.n_layers=n_layers
        self.output_size=output_size
        self.input_weights = torch.nn.Linear(input_size, hidden_dim)
        self.hidden_weights = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_size)
        self.t_dec = t_dec

    def forward(self, input, sigma_noise, hidden=None):
        if hidden is None:
            hidden = torch.randn(input.size(0),self.hidden_dim).to(input.device)
            #hidden = torch.zeros(input.size(0),self.hidden_dim).to(input.device)
        
        def recurrence(input, hidden):
            h_new = torch.relu(self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
            #h_new = torch.tanh(self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
            return h_new
        def recurrence_lin(input, hidden):
            h_new = (self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
            return h_new

        net_units = torch.zeros(input.size(0),input.size(1),self.hidden_dim)
        steps = range(input.size(1))
        for i in steps:
            hidden = recurrence(input[:,i], hidden)
            #hidden = recurrence_lin(input[:,i], hidden)
            net_units[:,i]=hidden
            
        hidden = hidden.detach()
        out = net_units[:,self.t_dec].contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        read_out_units = self.fc(net_units)
        return out, hidden, net_units, read_out_units
            









