import torch
import torch.nn as nn


class DVD_GCN(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, DD, X, act_linear=False, dropout_rate=0.5, featureless=True):
        super().__init__()
        self.description = ''
        self.D = D
        self.DT = DT
        self.V = V
        self.DD = DD
        self.X = X
        self.W1 = nn.Parameter(torch.Tensor(self.XD.shape[1], hl_size))
        self.W2 = nn.Parameter(torch.Tensor(self.XD.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(hl_size, num_classes))
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))

    def update_doc(self, hd, md):
        return self.W_self_doc.mm(hd) + md

    def update_word(self, hd, md):
        return self.W_self_word.mm(hd) + md

    def aggregate_doc(self):
        return 1

    def aggregate_word(self):
        return 1

    def gcn_layer(self):
        return 1

    def forward(self):

        HD = self.XD
        HV = self.XW

        agg_v = self.V.mm(HV.mm(self.WV2)) + self.VD.mm(HD.mm(self.WD2)) + self.V.mm(self.VD.mm(HD.mm(self.WD3)))
        update_v = self.W_self_v.mm(HV) + agg_v
        update_v = self.dropout(update_v)  # dropout before activation
        HV = self.act(update_v)

        # aggregation: document + word nodes
        agg_d = self.DD.mm(HD.mm(self.WD)) + self.DV.mm(HV.mm(self.WV))
        out = self.W_self_d.mm(HD) + agg_d  # no activation function on the last layer

        return out