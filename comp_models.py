import pickle
import sys

import numpy as np
from load_data import load
from utils import mkdir, normalize_adj_pw,AdjacencyMatrixCreatorForModularity
import torch
from agg_adj import *
from build_adj import build_adj_pmi_tfidf
import os
from pathlib import Path
from train_model import train_model
import pandas as pd
import json

def main(args):
    if args.model_dir is not None:
        mkdir(args.model_dir)
    if args.results_dir is not None:
        mkdir(args.results_dir)
    if args.adj_mat_dir is not None:
        mkdir(args.adj_mat_dir)

    # set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device

    # load data
    df_train, df_val, df_test, Y_train, Y_val, Y_test, train_size, val_size, test_size = load(args.dataset)

    # labels
    Y_train = torch.tensor(Y_train).to(torch.float32).to(device)
    Y_val = torch.tensor(Y_val).to(torch.float32).to(device)
    Y_test = torch.tensor(Y_test).to(torch.float32).to(device)
    
    # adjacency matrices based on PMI and TF-IDF
    adj_filename = os.path.join(args.adj_mat_dir, args.dataset + '_DW.pkl')
    if os.path.isfile(adj_filename):
        with open(adj_filename, "rb") as file:
            adj = pickle.load(file)
        #A = adj["A"]
        D = adj["D"].to(torch.float32).to(device)
        W = adj["W"].to(torch.float32).to(device)
        DT = adj["DT"].to(torch.float32).to(device)
    else:
        # build adjacency matrices based on PMI and TF-IDF
        D, W = build_adj_pmi_tfidf(df_train, df_val, df_test, args.dataset)
        D = D.to(torch.float32).to(device)
        W = W.to(torch.float32).to(device)
        DT = D.t()

        adj = {'D': D, 'DT': DT, 'W': W}
        with open(adj_filename, 'wb') as file:
            pickle.dump(adj, file)
    #Create adjacenncy matrix for modularity loss
    if args.graph_construction == 'matmul':
        creator = AdjacencyMatrixCreatorForModularity(method="from_symmetric")
        # mod_adj_matrix = creator.create(args=args, df_train=df_train, df_val=df_val, df_test=df_test) #symmetric matrix
        mod_adj_matrix = creator.create(doc_word_matrix=D, df_train=df_train, df_val=df_val, df_test=df_test, args=args)
        mod_adj_matrix = mod_adj_matrix.to(device)
    elif args.graph_construction == 'gaussian':
        creator = AdjacencyMatrixCreatorForModularity(method="from_sbert_embeddings_gaussian")
        # mod_adj_matrix = creator.create(args=args, df_train=df_train, df_val=df_val, df_test=df_test) #symmetric matrix
        mod_adj_matrix = creator.create(doc_word_matrix=D, df_train=df_train, df_val=df_val, df_test=df_test, args=args)
        mod_adj_matrix = mod_adj_matrix.to(device)
    elif args.graph_construction == 'cosine' and args.llm != 'gpt':
        creator = AdjacencyMatrixCreatorForModularity(method="from_sbert_embeddings")
        # mod_adj_matrix = creator.create(args=args, df_train=df_train, df_val=df_val, df_test=df_test) #symmetric matrix
        mod_adj_matrix = creator.create(doc_word_matrix=D, df_train=df_train, df_val=df_val, df_test=df_test, args=args)
        mod_adj_matrix = mod_adj_matrix.to(device)
    elif args.graph_construction == 'cosine' and args.llm == 'gpt':
        creator = AdjacencyMatrixCreatorForModularity(method="from_gpt_embeddings_cosine")
        # mod_adj_matrix = creator.create(args=args, df_train=df_train, df_val=df_val, df_test=df_test) #symmetric matrix
        mod_adj_matrix = creator.create(doc_word_matrix=D, df_train=df_train, df_val=df_val, df_test=df_test, args=args)
        mod_adj_matrix = mod_adj_matrix.to(device)
    # adjacency matrix normalization
    if args.normalization == 'sym':
        D_left = torch.sum(D, dim=1, keepdim=True).to_dense()
        D_right = torch.sum(D, dim=0, keepdim=True).to_dense()
        W_left = torch.sum(W, dim=0, keepdim=True).to_dense()
        W_right = W_left
        D_lpw = -0.5
        D_rpw = -0.5
        W_lpw = -0.5
        W_rpw = -0.5
    elif args.normalization == 'sym-dw':
        D_left = torch.sum(D, dim=1, keepdim=True).to_dense()
        D_right = torch.sum(D, dim=0, keepdim=True).to_dense() + torch.sum(W, dim=0, keepdim=True).to_dense()
        W_left = D_right
        W_right = W_left
        D_lpw = -0.5
        D_rpw = -0.5
        W_lpw = -0.5
        W_rpw = -0.5
    elif args.normalization == 'D-none-W-sym':
        D_left = torch.ones((1, D.shape[0]))
        D_right = torch.ones((1, D.shape[1]))
        W_left = torch.sum(W, dim=0, keepdim=True).to_dense()
        W_right = W_left
        D_lpw = 1
        D_rpw = 1
        W_lpw = -0.5
        W_rpw = -0.5
    elif args.normalization == 'D-sym-W-none':
        D_left = torch.sum(D, dim=1, keepdim=True).to_dense()
        D_right = torch.sum(D, dim=0, keepdim=True).to_dense()
        W_left = torch.ones((1, W.shape[0]))
        W_right = W_left
        D_lpw = -0.5
        D_rpw = -0.5
        W_lpw = 1
        W_rpw = 1
    elif args.normalization == 'D-row-W-sym':
        D_left = torch.sum(D, dim=1, keepdim=True).to_dense()
        D_right = torch.ones((1, D.shape[1]))
        W_left = torch.sum(W, dim=0, keepdim=True).to_dense()
        W_right = W_left
        D_lpw = -1
        D_rpw = 1
        W_lpw = -0.5
        W_rpw = -0.5
    elif args.normalization == 'D-sym-W-row':
        D_left = torch.sum(D, dim=1, keepdim=True).to_dense()
        D_right = torch.sum(D, dim=0, keepdim=True).to_dense()
        W_left = torch.sum(W, dim=0, keepdim=True).to_dense()
        W_right = torch.ones((1, W.shape[0]))
        D_lpw = -0.5
        D_rpw = -0.5
        W_lpw = -1
        W_rpw = 1
    else:
        os.error('{} - normalization method is undefined!', args.normalization)
    D = normalize_adj_pw(D, D_left, D_right, D_lpw, D_rpw)
    DT = D.t()
    W = normalize_adj_pw(W, W_left, W_right, W_lpw, W_rpw)

    num_classes = Y_train.shape[1]
    criterion = torch.nn.CrossEntropyLoss()  # define loss function

    if args.model == 'all':
        # [GCN_DVpDT_TG, GCN_DVpDT, GCN_DV, GCN_DDT, GCN_DVDT, GCN_DVDT_p1, GCN_DVDT_p2]
        models = [GCN_DVpDT_TG] #[GCN_DVpDT_TG, GCN_DV, GCN_DDT, GCN_DVDT, GCN_DVDT_p1, GCN_DVDT_p2]
    else:
        models = [getattr(sys.modules[__name__], args.model)]
    results = []
    for func in models:
        torch.manual_seed(args.seed)
        # initialize NN model
        model = func(num_classes, args.hl_size, D, DT, W, act_linear=args.act_linear, dropout_rate=args.dropout_rate).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        result = train_model(model, args, optimizer, D, criterion, Y_train, Y_val, Y_test, train_size, val_size, test_size, args.gamma, mod_adj_matrix)
        results.append(result)
    # Write validation F1s to result.txt for hyperparameter optimization
    # Handle different result files for ablation study
    if args.ablation_study:
        res_file = f"ablation_result_{args.dataset}_{args.graph_construction}_{args.modify_graph}_{args.label_type}.txt"
        save_dir = "results/ablation_gamma"
        # Write both validation and test metrics for ablation study
        result_metrics = results[0]
        
        # Get validation metrics
        val_micro = result_metrics.get('val_micro_f1', 0.0)
        val_macro = result_metrics.get('val_macro_f1', 0.0)
        
        # Get test metrics - they are stored with different keys in the results
        test_micro = result_metrics.get('micro_f1', 0.0)  # Test micro F1
        test_macro = result_metrics.get('macro_f1', 0.0)  # Test macro F1
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Write metrics to file
        with open(res_file, "w") as f:
            metric_line = f"{val_micro},{val_macro},{test_micro},{test_macro}"
            f.write(metric_line)
        
        # Debug print to verify values
        print(f"\nSaving metrics for {args.dataset}:")
        print(f"Validation - Micro: {val_micro:.4f}, Macro: {val_macro:.4f}")
        print(f"Test - Micro: {test_micro:.4f}, Macro: {test_macro:.4f}")
    else:
        res_file = f"result_{args.dataset}_{args.graph_construction}_{args.modify_graph}_{args.label_type}.txt"
        save_dir = "results"
        if args.hyperparameter_tuning == True:
            with open(res_file, "w") as f:
                f.write(f"{results[0]['val_micro_f1']},{results[0]['val_macro_f1']}")
        if args.hyperparameter_tuning == False:
            save_results(results, args)
    # print(df)

def save_results(results, args):
    # Build directory structure
    case_dir = f"{args.label_type}_{args.modify_graph}_{args.graph_construction}"
    # dataset_dir = args.dataset

    save_dir = os.path.join(args.results_dir, case_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Save results as CSV
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(save_dir, f"{args.dataset}_{args.seed}_results.csv"), index=False)

    # Save params as JSON
    params = vars(args)
    with open(os.path.join(save_dir, f"params_{args.dataset}.json"), "w") as f:
        json.dump(params, f, indent=4)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Adjacency Aggregator for Text Classification", add_help=add_help)
    parser.add_argument("--data-path", default="inputs/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="MR", type=str, help="dataset name")
    parser.add_argument("--model-dir", default=None, type=str, help="path to save trained models")
    parser.add_argument("--results-dir", default="results/", type=str, help="path to save results")
    parser.add_argument("--adj-mat-dir", default="adj_mat/", type=str, help="path to save adjacency matrices")
    parser.add_argument("--embeddings-dir", default="embeddings/", type=str, help="path to save embeddings")
    parser.add_argument("--epochs", default=200, type=int, help="number of total epochs to run")
    parser.add_argument("--hl_size", default=512, type=int, help="size of first hidden layer")
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
    parser.add_argument("--device", default="cpu", type=str, help="device to run the code")
    parser.add_argument("--act_linear", action='store_true', help="linear activation function")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--normalization", default='sym-dw', type=str, help="normalization method for adjacency matrix")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--patience", default=30, type=int, help="patience")
    parser.add_argument("--model", default='all', type=str, help="classification model")
    parser.add_argument("--gamma", default=1, type=float, help="resolution parameter")
    parser.add_argument("--mod_weight", default=1, type=float, help="weight of modularity parameter")
    parser.add_argument("--get_embeddings", default=False, help = "Store the embeddings")
    parser.add_argument("--mode", default="modularity", choices=["basemodel", "modularity", "lpa"], help = "Add the type of normalization")
    parser.add_argument("--increase", default=1, type=float, help = "Hyperparam for graph construction")
    parser.add_argument("--decrease", default=1, type=float, help = "Hyperparam for graph contruction")
    parser.add_argument("--k", default=10, type=int, help = "Number of similar nodes to consider")
    parser.add_argument("--graph_modification", dest="modify_graph", action="store_true")
    parser.set_defaults(modify_graph=False)
    parser.add_argument("--no-hyperparameter_tuning", dest="hyperparameter_tuning", action="store_false")
    parser.set_defaults(hyperparameter_tuning=True)
    parser.add_argument("--graph_construction", default="cosine", choices = ["cosine", "matmul", "gaussian"], help="choose the method of constructing the graph")
    parser.add_argument("--llm", type=str, help="Model to use for graph construction",default="all-mpnet-base-v2")
    parser.add_argument("--label_type", choices=["true_label","pred_label"], default="pred_label")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--ablation_study", action="store_true", help="Run as part of ablation study")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
