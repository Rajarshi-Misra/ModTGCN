import os, optuna, ast
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20ng', help='name of dataset')
parser.add_argument('--classifier', type=str, default='log_reg', choices=['log_reg', 'mlp'], help='type of classifier to use')
parser.add_argument('--tune', action='store_true', help='whether to perform hyperparameter tuning')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
def load_data(dataset: str):
    # Load embeddings
    df_train = np.load(f'../embeddings/sbert_fine_tuned/{dataset}/train_embeddings.npy')
    df_val   = np.load(f'../embeddings/sbert_fine_tuned/{dataset}/val_embeddings.npy')
    df_test  = np.load(f'../embeddings/sbert_fine_tuned/{dataset}/test_embeddings.npy')

    # Load labels
    y_train = np.load(f'../embeddings/sbert_fine_tuned/{dataset}/Y_train.npy', allow_pickle=True)
    y_val   = np.load(f'../embeddings/sbert_fine_tuned/{dataset}/Y_val.npy', allow_pickle=True)
    y_test  = np.load(f'../embeddings/sbert_fine_tuned/{dataset}/Y_test.npy', allow_pickle=True)

    # Sizes
    train_size = df_train.shape[0]
    val_size   = df_val.shape[0]
    test_size  = df_test.shape[0]

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}, {y_train.shape}")
    print("Data loaded successfully")

    return df_train, df_val, df_test, y_train, y_val, y_test

def train_classifier(X_train, y_train, X_val, y_val, seed, classifier='log_reg', params={}):
    random_state=seed
    if classifier == 'log_reg':
            clf = LogisticRegression(random_state=random_state, **params, n_jobs=-1, tol=1e-3)
    elif classifier == 'mlp':
            clf = MLPClassifier(random_state=random_state, max_iter=300,**params)
    else:
        raise ValueError("Invalid classifier type.")
    clf.fit(X_train, y_train)
    
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")  # macro F1 for class imbalance
    return val_acc, val_f1


def objective( args,trial, X_train, y_train, X_val, y_val, classifier='log_reg'):
    if classifier == 'log_reg':
        solver = trial.suggest_categorical("solver", ["liblinear", "saga", "lbfgs", "newton-cg"])

        # Conditional penalty choices based on solver
        if solver in ["lbfgs", "newton-cg"]:
            penalty = "l2"  # these solvers only support L2
        elif solver == "liblinear":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        elif solver == "saga":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        else:
            penalty = "l2"

        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2,log=True),
            "penalty": penalty,
            "solver": solver,
            "max_iter": 1000,
        }

    elif classifier == 'mlp':
        params = {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(64,), (128,), (256,), (128, 64), (256, 128)]
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),  # L2 reg strength
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        }
    else:
        raise ValueError("Invalid classifier type.")

    val_acc, val_f1 = train_classifier(X_train, y_train, X_val, y_val, args.seed , classifier, params)
    return val_acc, val_f1


def tune_with_optuna(args, X_train, y_train, X_val, y_val, classifier='log_reg', n_trials=30):
    seed = 42
    study = optuna.create_study(directions=["maximize","maximize"], sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective(args,trial, X_train, y_train, X_val, y_val, classifier), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trials[0]  # Best trial based on first objective (accuracy)
    print(f"  Accuracy: {trial.values[0]}")
    print(f"  F1 Score: {trial.values[1]}")
    print("  Params: ")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
    return study


def evaluate_best_model(best_params, X_train, y_train, X_test, y_test, seed, classifier='log_reg'):
    random_state = seed
    if classifier == 'log_reg':
        clf = LogisticRegression(random_state=random_state, **best_params)
    elif classifier == 'mlp':
        clf = MLPClassifier(random_state=random_state, max_iter=300, **best_params)
    else:
        raise ValueError("Invalid classifier type.")

    # Train on full data
    clf.fit(X_train, y_train)

    # Predict on test
    y_pred = clf.predict(X_test)

    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    print("Test Accuracy:", test_acc)
    print("Test Macro F1:", test_f1)

    return clf, test_acc, test_f1


def create_tsne_plot(df_train, df_val, df_test, Y_train, Y_val, Y_test, title="t-SNE visualization"):
    X = np.vstack([df_train, df_val, df_test])
    y = np.hstack([Y_train, Y_val, Y_test])
    os.makedirs("../tsne_plots", exist_ok=True)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1), max_iter=1000)
    tsne_result = tsne.fit_transform(df_test)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(Y_test)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        indices = Y_test == label
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                    c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
    
    plt.title(f't-SNE Visualization')
    plt.tight_layout()
    plt.savefig(f"../tsne_plots/{title}.png")
    plt.close()

def load_hyperparams(filepath):
    with open(filepath, 'r') as f:
        line = f.readline()
        params_str = line.split('Params: ')[-1].strip()
        params = ast.literal_eval(params_str)
    return params

def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    df_train, df_val, df_test, Y_train, Y_val, Y_test = load_data(args.dataset) 
    # create_tsne_plot(df_train, df_val, df_test,Y_train, Y_val, Y_test,
    #       title=f"t-SNE for test {args.dataset}")
    # clf = LogisticRegression()
    # clf.fit(df_train, Y_train)
    # y_pred_untuned = clf.predict(df_test)
    # untuned_acc = accuracy_score(Y_test, y_pred_untuned)
    # untuned_f1 = f1_score(Y_test, y_pred_untuned, average="macro")
    # print(f"Untuned {args.classifier} on {args.dataset} - Accuracy: {untuned_acc}, Macro F1: {untuned_f1}")

    if args.tune and args.seed == 0:
        study_lr = tune_with_optuna(args, df_train, Y_train, df_val, Y_val, classifier=args.classifier, n_trials=30)
        best_params = study_lr.best_trials
        print(f"Best hyperparameters for {args.classifier} on {args.dataset}: {best_params}")

        with open(f"../linear_results/{args.dataset}_{args.classifier}_best_hyperparams.txt", "w") as f:
            for t in study_lr.best_trials:
                f.write(f"Accuracy: {t.values[0]:.4f}, Macro F1: {t.values[1]:.4f}, Params: {t.params}\n")
    else:
        best_params = load_hyperparams(f'../linear_results/{args.dataset}_{args.classifier}_best_hyperparams.txt')
        clf_lr, acc_lr, f1_lr = evaluate_best_model(best_params, df_train, Y_train, df_test, Y_test, args.seed, classifier=args.classifier)
        print(f"Final results for {args.classifier} on {args.dataset} - Accuracy: {acc_lr}, Macro F1: {f1_lr}")
        with open(f"../linear_results/{args.dataset}_{args.classifier}.txt", "w") as f:
            f.write(f"Final results for {args.classifier} on {args.dataset} - Accuracy: {acc_lr}, Macro F1: {f1_lr}")

if __name__ == "__main__":
    main()
    
