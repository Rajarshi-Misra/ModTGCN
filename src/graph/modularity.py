import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.utils import gaussian_dist

class AdjacencyMatrixCreatorForModularity:
    """
    A class for creating adjacency matrices using different methods.
    """
    def __init__(self, method="from_symmetric"):
        """
        Initializes the AdjacencyMatrixCreator with the specified method.
        """
        self.method = method

    def create(self, args=None, df_train = None, df_test = None, df_val = None):
        """
        Creates the adjacency matrix using the specified method.
        """
        if self.method == "from_sbert_embeddings_cosine":
            return self._from_SBERT_embeddings_cosine(args, df_train, df_val, df_test)
        elif self.method == "from_sbert_embeddings_gaussian":
            return self._from_SBERT_embeddings_gaussian(args, df_train, df_val, df_test)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _from_SBERT_embeddings_cosine(self, args, df_train, df_val, df_test):
        documents = np.concatenate([df_train['text'].values, df_val['text'].values, df_test['text'].values])
        sbert_model = SentenceTransformer(args.llm)
        embeddings = sbert_model.encode(documents, convert_to_tensor=True)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # [N, D]
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        if args.modify_graph:
            total_size = sim_matrix.size(0)
            adj_matrix = torch.zeros_like(sim_matrix, dtype=torch.float32)

            train_labels = df_train['label'].values
            train_size = len(train_labels)

            _, top_k_indices = torch.topk(sim_matrix, args.k, dim=1)

            for i in range(total_size):
                for j in top_k_indices[i]:
                    if i == j:  
                        continue
                    sim = sim_matrix[i, j]
                    if i < train_size and j < train_size:
                        if train_labels[i] == train_labels[j]:
                            sim *= args.increase
                        else:
                            sim *= args.decrease
                    adj_matrix[i, j] = sim
                    adj_matrix[j, i] = sim
            return adj_matrix

        return sim_matrix
    
    def _from_SBERT_embeddings_gaussian(self, args, df_train, df_val, df_test):
        documents = np.concatenate([df_train['text'].values, df_val['text'].values, df_test['text'].values])
        sbert_model = SentenceTransformer(args.llm)
        embeddings = sbert_model.encode(documents, convert_to_tensor=True)
        sim_matrix = gaussian_dist(embeddings, args.sigma)
        if args.modify_graph:
            total_size = sim_matrix.size(0)
            adj_matrix = torch.zeros_like(sim_matrix, dtype=torch.float32)

            train_labels = df_train['label'].values
            train_size = len(train_labels)

            _, top_k_indices = torch.topk(sim_matrix, args.k, dim=1)

            for i in range(total_size):
                for j in top_k_indices[i]:
                    if i == j:  
                        continue
                    sim = sim_matrix[i, j]
                    if i < train_size and j < train_size:
                        if train_labels[i] == train_labels[j]:
                            sim *= args.increase
                        else:
                            sim *= args.decrease
                    adj_matrix[i, j] = sim
                    adj_matrix[j, i] = sim

            return adj_matrix
        return sim_matrix