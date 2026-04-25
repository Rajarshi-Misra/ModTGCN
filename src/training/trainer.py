import torch
import torch.nn.functional as F

class ModularityTrainer:
    def __init__(self, model, optimizer, criterion, args, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.device = device
        self.best_metrics = {}
        self.history = []
    
    def prepare_modularity(self, doc_doc_adj):
        degree_W = torch.sum(doc_doc_adjdim=0)
        e_W = (torch.sum(doc_doc_adj)/2)
        modularity_matrix = (doc_doc_adj  - torch.outer(degree_W,degree_W)/(2*e_W))
        return modularity_matrix
    
    def train_epoch(self, Y_train, Y_val, Y_test, train_size, val_size, test_size, modularity_matrix, edge_weight):
        self.model.train()
        self.optimizer.zero_grad()
        _, outputs = self.model()
        loss_train = self.criterion(outputs[0: train_size], Y_train.float())
        soft_labels = F.softmax(outputs,dim=1)
        pred_labels_true = torch.cat([
            Y_train,
            soft_labels[train_size:]
        ], dim=0)
        ## TODO: INSTEAD OF ALWAYS CALCULATING MODULARITY... MAKE IT PLUGGABLE
        y_pred = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
        label_for_modularity = soft_labels if self.args.label_type=="pred_label" else pred_labels_true
        loss_mod_W = torch.trace(torch.matmul(torch.matmul(label_for_modularity.T, self.args.gamma*modularity_matrix), label_for_modularity)) * (1/(2*edge_weight))
        loss = (1-self.args.mod_weight)*loss_train - self.args.mod_weight*(loss_mod_W) if self.args.mode in ("modularity") else loss_train
        loss.backward()
        self.optimizer.step()
    
    # def fit(self, Y_train, Y_val, Y_test, train_size, val_size, test)