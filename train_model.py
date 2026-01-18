import torch
import time
from torcheval.metrics.functional import multiclass_f1_score
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))

def train_model(model, args, optimizer, D, criterion, Y_train, Y_val, Y_test, train_size, val_size, test_size, gamma, A_W):
    max_val_acc = 0
    counter = 0
    A_D = D.to_dense() if D.is_sparse else D
    
    D_W = torch.sum(A_W,dim=0).to(args.device)
    e_W = (torch.sum(A_W)/2).to(args.device)
    B_W = (A_W  - torch.outer(D_W,D_W)/(2*e_W)).to(args.device)
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