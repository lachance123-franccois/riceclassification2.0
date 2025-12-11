import torch.nn as nn

class monModel(nn.Module):
    def __init__(self,input_dim,nbr_hidden,):
        super(monModel,self).__init__()
        self.input_layer=nn.Linear(input_dim,nbr_hidden)
        self.output_layer=nn.Linear(nbr_hidden,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.input_layer(x)
        x=self.output_layer(x)
        x=self.sigmoid(x)
        return x