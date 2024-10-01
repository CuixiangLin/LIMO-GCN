import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import pandas as pd
from sklearn.linear_model import Ridge
class LIMOGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(LIMOGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.fc1 = nn.Linear(nfeat,nhid)
        self.fc2 = nn.Linear(nhid,nclass)
        self.dropout = dropout
    def forward(self, x, adj,a):
        # linear part
        y1=0
        if a!=0:
            x1 = self.fc1(x)
            y1 = self.fc2(x1)
            y1 = F.log_softmax(y1, dim=1)
        # GCN part
        y2=0
        if (a-1)!=0:
            x2 = F.relu(self.gc1(x, adj))
            x2 = F.dropout(x2, self.dropout, training=self.training)
            y2 = self.gc2(x2, adj)
            y2 = F.log_softmax(y2, dim=1)
        y = a*y1+(1-a)*y2
        return y




