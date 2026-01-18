import torch
import torch.nn as nn
import torch.nn.functional as F

class agg_adj(nn.Module):
    def __init__(self, num_classes, hl_size, *A_matrices, device=None):
        super().__init__()
        self.A_matrices = A_matrices
        self.W_matrices = nn.ParameterList([nn.Parameter(torch.Tensor(A.shape[1], hl_size)) for A in A_matrices])
        self.W_out = nn.Parameter(torch.Tensor(hl_size, num_classes))
        self.act = nn.ReLU(True)
        self.num_adj = len(A_matrices)
        self.dropout = nn.Dropout(p=0.0)

        for W in self.W_matrices:
            torch.nn.init.xavier_uniform_(W, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W_out, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        A_matrices = [self.dropout(A) for A in self.A_matrices]

        # Compute the product of each adjacency matrix with its corresponding weight matrix
        P_matrices = [self.act(torch.matmul(A, W)) for A, W in zip(A_matrices, self.W_matrices)]

        # Sum the results
        P_sum = sum(P_matrices)

        P_sum = self.dropout(P_sum)

        # max operation
        # P = torch.stack(P_matrices, dim=0)
        # P_sum = torch.max(P, 0).values

        # Compute the final output
        out = torch.matmul(P_sum, self.W_out)
        return out


class agg_adj_hl2(nn.Module):

    def __init__(self, num_classes, hl_size, *A_matrices, device=None):
        super().__init__()
        self.A_matrices = A_matrices
        self.W1_matrices = nn.ParameterList([nn.Parameter(torch.Tensor(A.shape[1], hl_size)) for A in A_matrices])
        self.W2_matrices = nn.ParameterList([nn.Parameter(torch.Tensor(hl_size, int(hl_size / 2))) for A in A_matrices])
        self.W_out = nn.Parameter(torch.Tensor(int(hl_size / 2), num_classes))
        self.act = nn.ReLU(True)

        for W in self.W1_matrices:
            torch.nn.init.xavier_uniform_(W, gain=nn.init.calculate_gain('relu'))
        for W in self.W2_matrices:
            torch.nn.init.xavier_uniform_(W, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W_out, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        # Compute the product of each adjacency matrix with its corresponding weight matrix
        P1 = [self.act(torch.matmul(A, W)) for A, W in zip(self.A_matrices, self.W1_matrices)]

        P2 = [self.act(torch.matmul(A, W)) for A, W in zip(P1, self.W2_matrices)]

        # Sum the results
        P_sum = sum(P2)

        # Compute the final output
        out = torch.matmul(P_sum, self.W_out)
        return out


class textGCN(nn.Module):
    def __init__(self, num_classes, hl_size, A, device=None):
        super().__init__()
        #self.F = torch.ones((A.shape[0], ))
        self.A = A
        self.W = nn.Parameter(torch.Tensor(A.shape[1], hl_size))
        self.W_out = nn.Parameter(torch.Tensor(hl_size, num_classes))
        self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        torch.nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W_out, gain=nn.init.calculate_gain('linear'))

    def forward(self):

        # Compute the product of each adjacency matrix with its corresponding weight matrix
        P = self.act(torch.matmul(self.A, self.W))
        #P = self.dropout(P)

        # Compute the final output
        out = torch.matmul(self.A, torch.matmul(P, self.W_out))
        return out


# softmax(D phi(V W) W)
class GCN_DV(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D (phi(V W1)) Wo)'
        self.D = D
        self.V = V
        self.W1 = nn.Parameter(torch.Tensor(V.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(hl_size, num_classes))
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        #W1 = self.dropout(self.W1)
        P = self.act(torch.matmul(self.V, self.W1))
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))
        return out


class GCN_DV2(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D (phi(V phi(V W1) W2)) Wo)'
        self.D = D
        self.V = V
        self.W1 = nn.Parameter(torch.Tensor(V.shape[1], hl_size))
        self.W2 = nn.Parameter(torch.Tensor(hl_size, int(hl_size / 2)))
        self.Wo = nn.Parameter(torch.Tensor(int(hl_size / 2), num_classes))
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        P = self.act(torch.matmul(self.V, self.W1))
        P = self.act(torch.matmul(torch.matmul(self.V, P), self.W2))
        P = self.dropout(P)
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))
        return out


# softmax(D phi(DT W) W)
class GCN_DDT(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D (phi(DT W1)) Wo)'
        self.D = D
        self.DT = DT
        self.W1 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(hl_size, num_classes))
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        P = self.act(torch.matmul(self.DT, self.W1))
        P = self.dropout(P)
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))
        return out


class GCN_DVpDT(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D (phi(V W1) + phi(DT W2))) Wo)'
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
        P = self.act(torch.matmul(self.V, self.W1)) + self.act(torch.matmul(self.DT, self.W2))
        P = self.dropout(P)
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))
        return out


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

class GCN_DVpDT_TG_att(nn.Module): 
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5, alpha = 0.5):
        super().__init__()
        self.description = 'softmax(D phi(V W1 + DT W2)) Wo)'
        self.D = D
        self.DT = DT
        self.V = V
        self.W1 = nn.Parameter(torch.Tensor(self.V.shape[1], hl_size))
        self.W2 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(hl_size, num_classes))
        
        self.alpha = alpha  # Interpolation factor for TF-IDF and attention outputs

        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))


    def forward(self):
        # Combine word-word and document-word projections
        P = self.act(torch.matmul(self.V, self.W1) + torch.matmul(self.DT, self.W2))
        P = self.dropout(P)

        tfidf_out = torch.matmul(self.D, torch.matmul(P, self.Wo))  # (N x C)
        attention = F.softmax(torch.sparse.mm(self.D, self.V.to_dense()), dim=1)

        # Final classification
        att_out = torch.matmul(torch.matmul(attention,P), self.Wo) 
        out = self.alpha * tfidf_out + (1 - self.alpha) * att_out
        doc_repr = self.alpha * torch.matmul(self.D, P) + (1-self.alpha) * torch.matmul(attention, P)
        return doc_repr, out


class GCN_DVpDT_TG_qkv(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5, alpha=0.5):
        super().__init__()
        self.description = 'Document-conditioned attention with TF-IDF interpolation'
        self.D = D               # (N x V)
        self.DT = DT             # (V x N)
        self.V = V               # (V x V)
        
        self.W1 = nn.Parameter(torch.Tensor(self.V.shape[1], hl_size))
        self.W2 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(hl_size, num_classes))

        self.doc_encoder = nn.Linear(self.D.shape[1], hl_size)
        self.W_att = nn.Parameter(torch.Tensor(hl_size, hl_size))

        if act_linear:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))
        torch.nn.init.xavier_uniform_(self.W_att, gain=nn.init.calculate_gain('linear'))

        self.alpha = alpha  # Interpolation factor for TF-IDF and attention outputs

    def forward(self):
        # Project word embeddings
        P = self.act(torch.matmul(self.V, self.W1) + torch.matmul(self.DT, self.W2))  # (V x H)
        P = self.dropout(P)

        # TF-IDF weighted output
        E_tfidf = torch.matmul(self.D, P)                # (N x H)
        tfidf_out = torch.matmul(E_tfidf, self.Wo)       # (N x C)

        if self.D.is_sparse:
            doc_repr = torch.sparse.mm(self.D.float(), P)
            doc_norms = torch.sparse.sum(self.D, dim=1).to_dense().unsqueeze(1)
            doc_norms = torch.clamp(doc_norms, min=1e-8)
            E_tfidf = doc_repr / doc_norms
            tfidf_out = torch.matmul(E_tfidf, self.Wo)
            # No attention path defined here — for sparse case, you may skip or implement sparse-safe attention
            return E_tfidf, tfidf_out
        else:
            # Dense: compute attention
            D_dense = self.D.float()
            D_encoded = self.act(self.doc_encoder(D_dense))  # (N x H)
            att_scores = torch.matmul(torch.matmul(D_encoded, self.W_att), P.T)  # (N x V)

            # Masking: zero out words not in the doc
            masked_att_scores = att_scores.masked_fill(D_dense == 0, -1e9)
            attn_weights = F.softmax(masked_att_scores, dim=1)  # (N x V)

            E_attn = torch.matmul(attn_weights, P)  # (N x H)
            attn_out = torch.matmul(E_attn, self.Wo)  # (N x C)

            # Interpolate both embeddings
            doc_repr = self.alpha * E_tfidf + (1 - self.alpha) * E_attn
            out = self.alpha * tfidf_out + (1 - self.alpha) * attn_out

            return doc_repr, out

"""
class GCN_DVpDT_TG_qkv(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'Document-conditioned attention for each word'
        self.D = D
        self.DT = DT
        self.V = V
        self.W1 = nn.Parameter(torch.Tensor(self.V.shape[1], hl_size))
        self.W2 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(hl_size, num_classes))
        
        # Document encoding parameters
        self.doc_encoder = nn.Linear(self.D.shape[1], hl_size)
        
        # Attention parameters - bilinear attention
        self.W_att = nn.Parameter(torch.Tensor(hl_size, hl_size))
        
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize parameters
        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))
        torch.nn.init.xavier_uniform_(self.W_att, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        # Word representations: P shape (vocab, hl_size)
        P = self.act(torch.matmul(self.V, self.W1) + torch.matmul(self.DT, self.W2))
        P = self.dropout(P)

        # For sparse tensors, we need to work with them directly
        if self.D.is_sparse:
            # Use sparse matrix multiplication directly
            # doc_repr shape: (n_docs, hl_size)
            doc_repr = torch.sparse.mm(self.D.float(), P)
            
            # For attention, we compute attention scores and apply them
            # Get document tf-idf sums for normalization
            doc_norms = torch.sparse.sum(self.D, dim=1).to_dense().unsqueeze(1)  # shape: (n_docs, 1)
            doc_norms = torch.clamp(doc_norms, min=1e-8)  # avoid division by zero
            
            # Normalize document representations
            doc_repr = doc_repr / doc_norms
            
        else:
            # Dense tensor operations
            D_dense = self.D.float()
            
            # Document representations for attention
            D_encoded = self.doc_encoder(D_dense)
            D_encoded = self.act(D_encoded)
            
            # Bilinear attention: each document attends to each word differently
            att_scores = torch.matmul(torch.matmul(D_encoded, self.W_att), P.T)
            
            # Mask attention scores - only attend to words present in document
            masked_att_scores = att_scores * D_dense
            masked_att_scores = masked_att_scores + (D_dense == 0) * (-1e9)
            
            # Apply softmax to get attention weights per document
            attn_weights = torch.softmax(masked_att_scores, dim=1)
            
            # Apply attention weights to get document representations
            doc_repr = torch.matmul(attn_weights, P)
    
        out = torch.matmul(doc_repr, self.Wo)
    
        return doc_repr, out

"""        
class GCN_DVDT(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D phi(V phi(DT W1)) Wo)'
        self.D = D
        self.DT = DT
        self.V = V
        self.w = 1  # nn.Parameter(torch.Tensor(1, 1))  # regularization parameter
        self.W1 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        # self.W2 = nn.Parameter(torch.Tensor(hl_size, int(hl_size/2)))
        self.Wo = nn.Parameter(torch.Tensor(int(hl_size), num_classes))
        #self.bias = nn.Parameter(torch.Tensor(1, 1))
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        # torch.nn.init.constant_(self.w, 1e-1)
        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        #torch.nn.init.xavier_uniform_(self.W1, gain=100)
        # torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        #DTW = self.act(torch.matmul(self.DT, self.W1))
        # VW = self.act(torch.matmul(torch.matmul(self.V, DTW), self.W2))

        DTW = torch.matmul(self.DT, self.W1)
        P = self.act(torch.matmul(self.V, DTW))
        P = self.dropout(P)
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))

        return out  #, torch.norm(DTW.detach()), torch.norm(P.detach())


class GCN_DVDT_p1(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D phi(V phi(DT W1)) + phi(DT W2)) Wo)'
        self.D = D
        self.DT = DT
        self.V = V
        self.W1 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
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
        VDTW = self.act(self.V.mm(self.DT.mm(self.W1)))
        DTW = self.act(self.DT.mm(self.W2))
        P = torch.cat((VDTW, DTW), dim=1)
        #DTW = torch.matmul(self.DT, self.W1)
        #P = self.act(torch.matmul(self.V, DTW)) + self.act(torch.matmul(self.DT, self.W2))
        P = self.dropout(P)
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))
        return out

class GCN_DVDT_p2(nn.Module):
    def __init__(self, num_classes, hl_size, D, DT, V, act_linear=False, dropout_rate=0.5):
        super().__init__()
        self.description = 'softmax(D phi(V DT W1 + DT W2) Wo)'
        self.D = D
        self.DT = DT
        self.V = V
        self.W1 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        self.W2 = nn.Parameter(torch.Tensor(self.DT.shape[1], hl_size))
        self.Wo = nn.Parameter(torch.Tensor(2*hl_size, num_classes))
        if act_linear is True:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('linear'))

    def forward(self):
        VDTW = self.V.mm(self.DT.mm(self.W1))
        DTW = self.DT.mm(self.W2)
        P = torch.cat((VDTW, DTW), dim=1)
        #P = torch.max(VDTW, DTW)
        # P = torch.matmul(self.V, DTW) + torch.matmul(self.DT, self.W2)
        P = self.dropout(P)
        P = self.act(P)

        #I1 = self.D.mm(P)
        #I2 = self.A.mm(self.WD)
        #max_out = torch.max(I1, I2, dim=1)
        #out = max_out.mm(self.W0)
        out = torch.matmul(self.D, torch.matmul(P, self.Wo))
        return out

