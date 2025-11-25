import os, optuna
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20ng', help='name of dataset')
parser.add_argument('--sigma', type=float, default=None, help='sigma in gaussian')
parser.add_argument('--k', type=float, default=None, help='top knn nodes to keep')

########################
# Load Data
########################
def load_data(dataset: str, device="cpu"):
    # Load embeddings
    df_train = torch.from_numpy(np.load(f'../embeddings/sbert_fine_tuned/{dataset}/train_embeddings.npy')).float().to(device)
    df_val   = torch.from_numpy(np.load(f'../embeddings/sbert_fine_tuned/{dataset}/val_embeddings.npy')).float().to(device)
    df_test  = torch.from_numpy(np.load(f'../embeddings/sbert_fine_tuned/{dataset}/test_embeddings.npy')).float().to(device)

    # Load labels
    y_train = torch.from_numpy(np.load(f'../embeddings/sbert_fine_tuned/{dataset}/Y_train.npy', allow_pickle=True)).long().to(device)
    y_val   = torch.from_numpy(np.load(f'../embeddings/sbert_fine_tuned/{dataset}/Y_val.npy', allow_pickle=True)).long().to(device)
    y_test  = torch.from_numpy(np.load(f'../embeddings/sbert_fine_tuned/{dataset}/Y_test.npy', allow_pickle=True)).long().to(device)

    # Merge into single embedding + labels
    X = torch.cat([df_train, df_val, df_test], dim=0)
    Y = torch.cat([y_train, y_val, y_test], dim=0)

    # Index splits
    idx_train = torch.arange(0, df_train.size(0), device=device)
    idx_val   = torch.arange(df_train.size(0), df_train.size(0)+df_val.size(0), device=device)
    idx_test  = torch.arange(df_train.size(0)+df_val.size(0), X.size(0), device=device)

    print(f"Data loaded: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
    return X, Y, idx_train, idx_val, idx_test

########################
# Build Adjacency (Gaussian Kernel)
########################
def normalize_adj(adj):
    adj = adj.to_dense() if hasattr(adj, 'to_dense') else adj
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    ret = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return ret

def build_adj(embeddings, sigma):
    """ Gaussian kernel similarity adjacency """
    dist = torch.cdist(embeddings, embeddings, p=2) ** 2  # squared Euclidean
    adj = torch.exp(-dist / (2 * sigma ** 2))
    # print(torch.allclose(adj, adj.T))
    adj.fill_diagonal_(1)  # self-loops
    norm_adj = normalize_adj(adj)
    mask = norm_adj<0.5
    norm_adj[mask] = 0
    print("Nan in norm",torch.isnan(norm_adj).any())
    return norm_adj

########################
# GCN Model
########################
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
# Two layer GCN
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    

def train_model(model, optimizer, features, adj, labels, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    y_true_train = labels[idx_train].cpu().numpy()
    y_pred_train = output[idx_train].detach().cpu().argmax(dim=1).numpy()
    acc_train = accuracy_score(y_true_train, y_pred_train)
    f1_train = f1_score(y_true_train, y_pred_train, average='macro')
    loss_train.backward()
    optimizer.step()

    # print(idx_val)
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    y_true_val = labels[idx_val].cpu().numpy()
    y_pred_val = output[idx_val].detach().cpu().argmax(dim=1).numpy()
    # print("y_true_val", y_true_val[-20:], "y_pred_val", y_pred_val[-20:])
    acc_val = accuracy_score(y_true_val, y_pred_val)
    f1_val = f1_score(y_true_val, y_pred_val, average='macro')
    # if epoch % 10 == 0:
    #     print('Epoch: {:04d}'.format(epoch+1),
    #         'loss_train: {:.4f}'.format(loss_train.item()),
    #         'acc_train: {:.4f}'.format(acc_train),
    #         'f1_train: {:.4f}'.format(f1_train),
    #         'loss_val: {:.4f}'.format(loss_val.item()),
    #         'acc_val: {:.4f}'.format(acc_val),
    #         'f1_val: {:.4f}'.format(f1_val),
    #         'time: {:.4f}s'.format(time.time() - t))

    return loss_train.item(), acc_train, f1_train, loss_val.item(), acc_val, f1_val


def objective(trial, X, labels, idx_train, idx_val, n_classes, device, graph_type):
    hidden = trial.suggest_categorical("hidden", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
    
    sigma = trial.suggest_float("sigma", 0.01, 10)
    k = trial.suggest_int("k", 8, 30, step=1)
    adj = build_knn_adj(X, k=k, device=device)
    features = torch.eye(X.shape[0]).to(device)
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=n_classes,
                dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_f1 = 0.0
    # train for fixed epochs and track best val F1
    for epoch in range(300):
        loss_train, acc_train, f1_train, loss_val, acc_val, f1_val = train_model(
            model, optimizer, features, adj, labels, idx_train, idx_val
        )
        best_val_f1 = max(best_val_f1, f1_val)
    return best_val_f1

def run_optuna(X, labels, idx_train, idx_val, n_classes, device, graph_type, n_trials=100):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, labels, idx_train, idx_val, n_classes, device, graph_type),
                   n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Val Accuracy: {trial.value:.4f}")
    print("  Params: ")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    return study

def test_model(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    y_true_test = labels[idx_test].cpu().numpy()
    y_pred_test = output[idx_test].detach().cpu().argmax(dim=1).numpy()
    print("y_true_test", y_true_test, "y_pred_test", y_pred_test[-20:])
    acc_test = accuracy_score(y_pred_test, y_true_test)
    f1_test = f1_score(y_pred_test, y_true_test, average='macro')
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test),
    #       "f1_score= {:.4f}".format(f1_test))
    return loss_test.item(), acc_test, f1_test

def build_knn_adj(embeddings, k=1, device="cpu"):
    """
    Build adjacency using cosine similarity + kNN.
    - embeddings: [N, d] tensor (node features, e.g. SBERT embeddings)
    - k: number of neighbors to keep per node
    """
    # normalize for cosine similarity
    emb_norm = F.normalize(embeddings, p=2, dim=1)  # [N, d]

    # cosine similarity matrix
    sim = torch.mm(emb_norm, emb_norm.t())  # [N, N], values in [-1,1]

    # keep top-k neighbors (excluding self)
    N = sim.size(0)
    topk_vals, topk_idx = torch.topk(sim, k=k+1, dim=1)  # k+1 because self will be included
    mask = torch.zeros_like(sim, dtype=torch.bool, device=device)

    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(topk_idx)
    mask[row_idx, topk_idx] = True
    mask.fill_diagonal_(False)  # remove self from mask (we’ll add later)

    # sparse adjacency: keep similarities for top-k neighbors
    adj = torch.zeros_like(sim)
    adj[mask] = sim[mask]

    # symmetrize (kNN is directed, make it undirected)
    adj = 0.5 * (adj + adj.t())

    # add self-loops
    adj.fill_diagonal_(1.0)

    # normalize adjacency like in GCN
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return norm_adj.to(device)

def build_gaussian_adj(embeddings, sigma=1.0, device="cpu"):
    """
    Build adjacency using Gaussian kernel without pruning.
    - embeddings: [N, d] tensor
    - sigma: Gaussian kernel width
    """
    # pairwise squared Euclidean distances
    dist = torch.cdist(embeddings, embeddings, p=2) ** 2  # [N, N]

    # Gaussian kernel similarity
    adj = torch.exp(-dist / (2 * sigma ** 2))  # [N, N]
    
    # add self-loops
    adj.fill_diagonal_(1.0)

    # normalize adjacency (GCN-style)
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return norm_adj.to(device)

def build_cosine_adj(embeddings, device="cpu"):
    """
    Build adjacency using cosine similarity without pruning.
    - embeddings: [N, d] tensor
    """
    # normalize for cosine similarity
    emb_norm = F.normalize(embeddings, p=2, dim=1)  # [N, d]

    # cosine similarity matrix
    adj = torch.mm(emb_norm, emb_norm.t())  # [N, N], values in [-1,1]
    
    # shift from [-1,1] to [0,1] range
    adj = (adj + 1) / 2
    
    # add self-loops
    adj.fill_diagonal_(1.0)

    # normalize adjacency (GCN-style)
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return norm_adj.to(device)

def build_gaussian_knn_adj(embeddings, sigma=1.0, k=10, device="cpu"):
    """
    Build adjacency using Gaussian kernel + kNN pruning.
    - embeddings: [N, d] tensor
    - sigma: Gaussian kernel width
    - k: number of neighbors per node (excluding self)
    """
    # pairwise squared Euclidean distances
    dist = torch.cdist(embeddings, embeddings, p=2) ** 2  # [N, N]

    # Gaussian kernel similarity
    adj = torch.exp(-dist / (2 * sigma ** 2))  # [N, N]

    # keep only top-k neighbors
    N = adj.size(0)
    topk_vals, topk_idx = torch.topk(adj, k=k+1, dim=1)  # include self
    mask = torch.zeros_like(adj, dtype=torch.bool, device=device)

    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(topk_idx)
    mask[row_idx, topk_idx] = True
    mask.fill_diagonal_(False)  # remove self before adding explicitly

    # sparse adjacency: keep similarities for top-k neighbors
    adj_pruned = torch.zeros_like(adj)
    adj_pruned[mask] = adj[mask]

    # symmetrize
    adj_pruned = 0.5 * (adj_pruned + adj_pruned.t())

    # add self-loops
    adj_pruned.fill_diagonal_(1.0) ##TODO: Try by removing self-loops

    # normalize adjacency (GCN-style)
    rowsum = torch.sum(adj_pruned, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj_pruned @ d_mat_inv_sqrt

    return norm_adj.to(device)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    X, Y, idx_train, idx_val, idx_test = load_data(args.dataset, device=device)
    features = torch.eye(X.shape[0]).to(device)
    
    # Define graph types to test
    graph_types = ['gaussian']
    
    for graph_type in graph_types:
        print(f"\nRunning experiments with {graph_type} similarity graph")

        # Run Optuna once with the specific graph type
        study = run_optuna(
            X, Y, idx_train, idx_val,
            n_classes=Y.max().item() + 1,
            device=device,
            graph_type=graph_type,
            n_trials=100
        )
        best_params = study.best_trial.params
        print("Best params found by Optuna:", best_params)

        with open(f"hyperparams_gcn_{args.dataset}.txt", "w") as f:
            f.write(str(best_params))

        # Run multiple seeds with best params
        seeds = [0, 1, 2, 3, 4]
        test_results = []

        for seed in seeds:
            print(f"\nRunning with seed {seed}")
            set_seed(seed)

            # Build adjacency matrix based on graph type
            adj = build_gaussian_knn_adj(X, k=best_params["k"], device=device)

            model = GCN(
                nfeat=features.shape[1],
                nhid=best_params["hidden"],
                nclass=Y.max().item() + 1,
                dropout=best_params["dropout"]
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"]
            )

            # Training loop
            epochs = 300
            for epoch in range(epochs):
                loss_train, acc_train, f1_train, loss_val, acc_val, f1_val = train_model(
                    model, optimizer, features, adj, Y, idx_train, idx_val
                )
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1:04d} | "
                        f"train_loss {loss_train:.4f} train_acc {acc_train:.4f} train_f1 {f1_train:.4f} "
                        f"| val_loss {loss_val:.4f} val_acc {acc_val:.4f} val_f1 {f1_val:.4f}")

            # Final test
            loss_test, acc_test, f1_test = test_model(model, features, adj, Y, idx_test)
            test_results.append([loss_test, acc_test, f1_test])
            print(f"Seed {seed} | Test loss={loss_test:.4f}, acc={acc_test:.4f}, f1={f1_test:.4f}")

        # Convert to numpy for easy averaging
        test_results = np.array(test_results)
        mean_results = test_results.mean(axis=0)
        std_results = test_results.std(axis=0)

        # Save results for this graph type
        result_str = (
            f"{graph_type} similarity results:\n"
            f"Test results across {len(seeds)} seeds\n"
            f"Mean: loss={mean_results[0]:.4f}, acc={mean_results[1]:.4f}, f1={mean_results[2]:.4f}\n"
            f"Std:  loss={std_results[0]:.4f}, acc={std_results[1]:.4f}, f1={std_results[2]:.4f}\n"
            f"Best parameters: {best_params}\n"
            f"-----------------------------------\n"
        )
        print(result_str)

        # Append results to file
        with open(f"../linear_results/results_graph_comparison_{args.dataset}.txt", "a") as f:
            f.write(result_str)

if __name__ == "__main__":
    main()