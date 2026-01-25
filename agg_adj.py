#TODO: RENAME IT AS MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_DVpDT_TG(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D phi(V W1 + DT W2)) Wo)'
        self.D = D
        self.DT = DT
        self.V = V
        self.W1 = nn.Parameter(torch.Tensor(self.V.shape[1], hl_size))
        self.W2 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(hl_size, num_classes))
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        P = self.act(torch.matmul(self.V, self.W1) + torch.matmul(self.DT, self.W2))
        P = self.dropout(P)
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))   
        return torch.matmul(self.D, P), out #embeddings, output