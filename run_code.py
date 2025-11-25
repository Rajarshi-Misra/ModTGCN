import os
import multiprocessing as mp
import ast
import sys

##TODO: FOR 20NG run it in sequence, and mention the combination in args

def run_command(command):
    os.system(command)

def load_hyperparams(filepath):
    with open(filepath, 'r') as f:
        line = f.readline()
        params_str = line.split('Params: ')[-1].strip()
        params = ast.literal_eval(params_str)
    return params

if __name__ == '__main__':
    # args: dataset graph_construction graph_weight_adj label_type
    dataset = sys.argv[1]
    graph_construction = sys.argv[2]   # e.g. gaussian / cosine / matmul
    graph_weight_adj = sys.argv[3]     # True / False
    label_type = sys.argv[4]           # true_label / pred_label

    device = 'cuda'
    epochs = 1000 
    hl_size = 200
    patience = 100
    NUM_SEEDS = 5

    # locate the correct hyperparameter file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"{label_type}_{graph_weight_adj}_{graph_construction}_{dataset}_hyperparams.txt"
    filepath = os.path.join(script_dir, "results", "best_hyperparams_fine_tuned", filename)
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Hyperparameter file not found: {filepath}")

    params = load_hyperparams(filepath)

    commands = []
    for seed in range(NUM_SEEDS):
        command = (
            f"python {os.path.join(script_dir, 'comp_models.py')} "
            f"--dataset {dataset} "
            f"--lr {params['lr']} "
            f"--epochs {epochs} "
            f"--hl_size {hl_size} "
            f"--device {device} "
            f"--seed {seed} "
            f"--normalization sym-dw "
            f"--dropout_rate {params['dropout_rate']} "
            f"--patience {patience} "
            f"--gamma {params['gamma']} "
            f"--mod_weight {params['mod_weight']} "
            f"--mode modularity "
            f"--get_embeddings False "
            f"--no-hyperparameter_tuning "
            f"--label_type {label_type} "
            f"--graph_construction {graph_construction} "
            f"--llm models/sbert/{dataset}/checkpoint "
        )
        if graph_weight_adj == "True":
            command += (
                f"--increase {params['increase']} "
                f"--decrease {params['decrease']} "
                f"--k {params['k']} "
                f"--graph_modification "
            )
        if graph_construction == "gaussian":
            command += f"--sigma {params['sigma']} "

        commands.append(command)

    if dataset == '20ng':
        for cmd in commands:
            run_command(cmd)
    else:
        with mp.Pool(processes=NUM_SEEDS) as pool:
            pool.map(run_command, commands)
    print("All runs finished.")
