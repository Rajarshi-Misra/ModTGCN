import torch
import time
from torcheval.metrics.functional import multiclass_f1_score
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parameters_norm(model):
    param_norm = [torch.norm(p.detach()) for p in model.parameters() if p.requires_grad]
    return param_norm

import torch

def compute_Qg(A: torch.Tensor, soft_labels: torch.Tensor, chi=1.0):
    m = A.sum() / 2
    H = soft_labels.T  # shape: (C, N)
    intra_c = torch.matmul(H, torch.matmul(A, H.T)).diag() / 2  # divide by 2 for undirected
    degrees = A.sum(dim=1)  # shape: (N,)
    K_c = torch.matmul(H, degrees)  # shape: (C,)
    n_c = soft_labels.sum(dim=0)  # shape: (C,)
    rho_c = 2 * intra_c / (n_c * (n_c - 1)+1e-10)
    Q_c = (2*intra_c - ((K_c**2) / (2 * m))) * (rho_c ** chi)
    Q_g = Q_c.sum() / (2 * m)  # divide by 2*m to normalize

    return Q_g

def straight_through_hard_assignment(soft_labels):
    hard_labels = F.one_hot(soft_labels.argmax(dim=1), num_classes=soft_labels.size(1)).float()
    return hard_labels - soft_labels.detach() + soft_labels

def calculate_rho(adj_mat, hard_labels,chi=1.0):
    num_classes = hard_labels.shape[1]
    class_indices = torch.argmax(hard_labels, dim=1)
    m = adj_mat.sum()
    Q_g = 0
    for cls in range(num_classes):
        nodes_in_class = (class_indices==cls).nonzero(as_tuple=True)[0]
        class_edges_sum = adj_mat[nodes_in_class][:, nodes_in_class].sum()
        degrees = len(nodes_in_class)*len(nodes_in_class-1)
        rho_cls = (2*class_edges_sum)/(num_classes*(num_classes-1))
        Q_g += (class_edges_sum - degrees**2/(2*m))*(rho_cls**chi)
    return Q_g/(2*m)


### train_model without rho
def train_model(model, args, optimizer, D, criterion, Y_train, Y_val, Y_test, train_size, val_size, test_size, gamma, A_W):
    # training_start_time = time.time()
    # training the model and report accuracy
    max_val_acc = 0
    counter = 0
    A_D = D.to_dense() if D.is_sparse else D
    
    D_W = torch.sum(A_W,dim=0).to(args.device)
    e_W = (torch.sum(A_W)/2).to(args.device)
    B_W = (A_W  - torch.outer(D_W,D_W)/(2*e_W)).to(args.device)
    # d_inv_sqrt = torch.pow(D_W, -0.5).to(args.device)
    # d_inv_sqrt[torch.isinf(d_inv_sqrt)]=0.0
    # D_inv_sqrt = torch.diag(d_inv_sqrt)
    # laplacian_matrix = torch.eye(A_W.size(0), device=A_W.device)-D_inv_sqrt@A_W@D_inv_sqrt
    mod_weight =  args.mod_weight
    modularity_per_epoch = []  # To store modularity values per epoch
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        model.train()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        embeddings, outputs = model()
        loss_train = criterion(outputs[0:train_size], Y_train.float()) ##cross entropy loss
        soft_labels = F.softmax(outputs,dim=1)
        pred_labels_true = soft_labels.clone()
        pred_labels_true = torch.cat([
            Y_train,                  # fixed one-hot for labeled data
            soft_labels[train_size:]  # soft labels for unlabeled
        ], dim=0)
        y_pred = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)##TODO: CHECK HOW DETACH WAS WORKING
        y_pred_float = torch.nn.functional.one_hot(y_pred, soft_labels.size(1)).float()
        label_for_modularity = soft_labels if args.label_type=="pred_label" else pred_labels_true
        loss_mod_W = torch.trace(torch.matmul(torch.matmul(label_for_modularity.T, gamma*B_W), label_for_modularity)) * (1/(2*e_W))
        # print("Mod Loss.....", loss_mod_W)
        # lpa_regularizer = torch.trace(torch.matmul(torch.matmul(y_pred_float.T, laplacian_matrix), y_pred_float))
        if args.mode == "modularity":
            regularizer = loss_mod_W
        # else:
        #     regularizer = -lpa_regularizer
        loss = (1-mod_weight)*loss_train - mod_weight*(regularizer) if args.mode in ("lpa","modularity") else loss_train
        loss.backward()
        optimizer.step()
        y_pred = y_pred[0:train_size+val_size+test_size]

        # Store modularity value for this epoch
        modularity_per_epoch.append(loss_mod_W.item())

        model.eval()
        val_loss = criterion(outputs[train_size:train_size + val_size], Y_val.float())

        train_acc = torch.sum(torch.argmax(Y_train, dim=1) == y_pred[:train_size]) / train_size
        val_acc = torch.sum(torch.argmax(Y_val, dim=1) == y_pred[train_size:train_size + val_size]) / val_size
        test_acc = torch.sum(torch.argmax(Y_test, dim=1) == y_pred[train_size + val_size:]) / test_size

        # scheduler.step(val_loss)

        num_classes = Y_train.shape[1]
        # Validation F1s
        val_macro_f1 = multiclass_f1_score(y_pred[train_size:train_size + val_size], torch.argmax(Y_val, dim=1),
                                           num_classes=num_classes, average='macro')
        val_micro_f1 = multiclass_f1_score(y_pred[train_size:train_size + val_size], torch.argmax(Y_val, dim=1),
                                           num_classes=num_classes, average='micro')
        # Test F1s (still computed for logging, but not used for optimization)
        test_macro_f1 = multiclass_f1_score(y_pred[train_size + val_size:], torch.argmax(Y_test, dim=1),
                                            num_classes=num_classes, average='macro')
        test_micro_f1 = multiclass_f1_score(y_pred[train_size + val_size:], torch.argmax(Y_test, dim=1),
                                            num_classes=num_classes, average='micro')
        class_f1 = multiclass_f1_score(y_pred[train_size + val_size:], torch.argmax(Y_test, dim=1),
                                            num_classes=num_classes, average=None)
        # class_f1_dict = {inverse_label_mapping[idx]: f1 for idx, f1 in enumerate(class_f1.cpu().numpy())}

        # print statistics
        if (epoch % 50) == 0:
            print(
                f'[{epoch + 1}] train loss: {loss.item():.9f}, val loss: {val_loss.item():.9f}, '
                f'train_acc:{train_acc * 100:5.2f}, val_acc:{val_acc * 100:5.2f}, test_acc:{test_acc * 100:5.2f}, '
                f'test_macro_f1:{test_macro_f1 * 100:5.2f}, test_micro_f1:{test_micro_f1 * 100:5.2f}, test_micro_f1:{class_f1} ')

        if max_val_acc < val_acc:
            max_val_acc = val_acc
            best_test_acc = test_acc
            best_acc_at_epoch = epoch + 1
            best_micro_f1 = test_micro_f1
            best_macro_f1 = test_macro_f1
            best_val_micro_f1 = val_micro_f1
            best_val_macro_f1 = val_macro_f1
            best_class_f1 = class_f1
            # best_f1_dict=class_f1_dict
            best_node_pred = y_pred
            best_embeddings = embeddings
            counter = 0
        else:
            counter = counter + 1
            if counter == args.patience:
                break

    
    print(f' best test accuracy {best_test_acc * 100:.2f}% at epoch {best_acc_at_epoch}.')

    # Plot modularity vs epoch
    try:
        plt.figure()
        plt.plot(range(1, len(modularity_per_epoch)+1), modularity_per_epoch, label=f"{args.dataset}")
        plt.xlabel("Epoch")
        plt.ylabel("Modularity (loss_mod_W)")
        plt.title(f"Modularity vs Epoch for {args.dataset}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path}/results/modularity_vs_epoch_{args.dataset}.png")
        plt.close()
    except ImportError:
        print("matplotlib not installed, skipping modularity plot.")

    # if args.get_embeddings:
    #     filename = f'{path}/embeddings/best_embeddings_mod_{args.dataset}.npy' if args.mode == "modularity" else f'best_embeddings_{args.dataset}.npy'
    #     np.save(filename, best_embeddings.detach().cpu().numpy())
    results = {'model': model.__class__.__name__, 'model_desc': model.description,
               'micro_f1': best_micro_f1.cpu().numpy() * 100,
               'macro_f1': best_macro_f1.cpu().numpy() * 100,
               'val_micro_f1': best_val_micro_f1.cpu().numpy() * 100,
               'val_macro_f1': best_val_macro_f1.cpu().numpy() * 100,
               'class_f1': best_class_f1.cpu().numpy(),
               'epoch': best_acc_at_epoch,
               'gamma': gamma
               }
    return results