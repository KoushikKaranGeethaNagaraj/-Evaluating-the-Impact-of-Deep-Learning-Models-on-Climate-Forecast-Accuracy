import torch
import torch.nn as nn

##TCN Architecture-Classic with causal convolution

#Things to keep in mind 
    # Temporal block is just one block with a specific padding and dilation(as padding and dilation matter
            #For causal convolution)
            #below temporal block is one of the basic TCN block
class TemporalBlockClassic(nn.Module):
    def __init__(self,input_n,output_n,kernel_size,padding,dilations,stride,dropout=0.2,batch_size=None,out_sequence=None):
        super(TemporalBlockClassic,self).__init__()
        #defining layers
        self.conv1=nn.Conv1d(in_channels=input_n,out_channels=input_n//4,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilations)
        self.trim1=TrimmingFuture(padding)
        self.relu1=nn.ReLU()
        self.bn1=nn.LayerNorm([input_n//4,out_sequence])
        self.dropout1=nn.Dropout(dropout)
        
        self.conv2=nn.Conv1d(in_channels=input_n//4,out_channels=input_n//2,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilations)
        self.trim2=TrimmingFuture(padding)
        self.relu2=nn.ReLU()
        self.bn2=nn.LayerNorm([input_n//2,out_sequence])
        self.dropout2=nn.Dropout(dropout)
        
        self.conv3=nn.Conv1d(in_channels=input_n//2,out_channels=input_n//4,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilations)
        self.trim3=TrimmingFuture(padding)
        self.relu3=nn.ReLU()
        self.bn3=nn.LayerNorm([input_n//4,out_sequence])
        self.dropout3=nn.Dropout(dropout)
        
        
        self.conv4=nn.Conv1d(in_channels=input_n//4,out_channels=output_n,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilations)
        self.trim4=TrimmingFuture(padding)
        self.relu4=nn.ReLU()
        self.bn4=nn.LayerNorm([output_n,out_sequence]) 
        self.dropout4=nn.Dropout(dropout)
        
        self.network=nn.Sequential(self.conv1,self.trim1,self.relu1,self.bn1,self.dropout1,
                                   self.conv2,self.trim2,self.relu2,self.bn2,self.dropout2,
                                   self.conv3,self.trim3,self.relu3,self.bn3,self.dropout3,
                                   self.conv4,self.trim4,self.relu4,self.bn4,self.dropout4)
        if input_n!=output_n:
            self.residual=nn.Conv1d(input_n,output_n,kernel_size=1)#maintains the input features
        else:self.residual=None
        self.relu_m=nn.ReLU()
        
    def forward(self,x):
        out=self.network(x)
        if self.residual==None:res=x
        if self.residual!=None:res=self.residual(x) 
        return self.relu_m(out+res)
    
#So during convolution padding is done on both sides and hence extra outputs are included using future 
# #that should be removed to maintain causal convolution
class TrimmingFuture(nn.Module):
    def __init__(self,trim_size):
        super(TrimmingFuture,self).__init__()
        self.trim_size=trim_size
    def forward(self,x):
        out=x[:,:,:-self.trim_size]
        return out
    
class ClassicTCNModel(nn.Module):
    def __init__(self,input_n,output_n,kernel_size,num_layers,dilations,batch_size,out_sequence,dropout=0.2,stride=1):
        super(ClassicTCNModel,self).__init__()
        #defining the tcn model
        layers=[]
        num_levels=len(num_layers)
        for i in range(num_levels):
            #most important as it decides the output size for each convolution
            padding=dilations[i]*(kernel_size-1)
            if i==0: 
                layers+=[TemporalBlockClassic(input_n=input_n,output_n=num_layers[i],kernel_size=kernel_size,padding=padding,dilations=dilations[i],stride=stride,dropout=dropout,batch_size=batch_size,out_sequence=out_sequence)]
            else:
                layers+=[TemporalBlockClassic(input_n=num_layers[i-1],output_n=num_layers[i],kernel_size=kernel_size,padding=padding,dilations=dilations[i],stride=stride,dropout=dropout,batch_size=batch_size,out_sequence=out_sequence)]
        
        self.combined_net=nn.Sequential(*layers)
        self.final=nn.Linear(num_layers[-1],output_n)
        
    def forward(self,x):
        x=x.permute(0,2,1)
        out=self.combined_net(x)
        out=self.final(out.permute(0,2,1))
        return out
            