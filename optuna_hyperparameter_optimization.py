import os
import optuna
import matplotlib.pyplot as plt
import pyttsx3
import optuna.visualization as vis
import sys

engine = pyttsx3.init()
# datasets = ['mr', 'r8', 'r52', 'ohsumed', '20ng']

graph_construction = sys.argv[1] ##cosine||gaussian||matrix multiplication(matmul)
graph_weight_adj = sys.argv[2] ##True or False
label_type = sys.argv[3] ## true_label || pred_label
dataset = sys.argv[4]
llm = sys.argv[5] if len(sys.argv) > 5 else f"models/sbert/{dataset}/checkpoint"

try:
    print(f"Starting study for dataset: {dataset}")

    # Reset result.txt to avoid contamination from previous dataset
    result_file = f"result_{dataset}_{graph_construction}_{graph_weight_adj}_{label_type}.txt"

    with open(result_file, "w") as f:
        f.write("0.0,0.0")  # Dummy initial value

    def objective(trial):
        # Suggest hyperparameters
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        normalization = 'sym-dw'
        lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
        gamma = trial.suggest_float('gamma', 0.1, 10)
        mod_weight = trial.suggest_float('mod_weight', 0.1, 0.9)

        seed = 0
        device = 'cuda'
        epochs = 300
        hl_size = 200
        patience = 100

        base_command = f"python comp_models.py --dataset {dataset} --lr {lr} --epochs {epochs} --hl_size {hl_size} --device {device} " \
                        f"--seed {seed } --normalization {normalization} --dropout_rate {dropout_rate} " \
                        f"--patience {patience} --gamma {gamma} --mod_weight {mod_weight} --mode modularity --graph_construction {graph_construction}  " \
                        f"--label_type {label_type} --llm {llm} " \

        # Build the command
        if graph_construction == "gaussian":
            sigma = trial.suggest_float('sigma', 0.1, 2.0)  # or your preferred range
            base_command += f"--sigma {sigma} " \
        
        if graph_weight_adj == "True":
            increase = trial.suggest_float('increase', 1.2, 2)
            decrease = trial.suggest_float('decrease', 0.4, 0.8)
            k = trial.suggest_int('k', 4, 15, step = 1)
            base_command += f"--graph_modification --increase {increase} --decrease {decrease} --k {k} " \

        # Run the command
        os.system(base_command)
        print("One run done")

        # Ensure result.txt is not empty before reading
        try:
            with open(result_file, "r") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Result file is empty!")
                val_micro_f1, val_macro_f1 = map(float, content.split(","))
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
            raise

        return val_micro_f1, val_macro_f1

    # Create and optimize study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        pruner=pruner
    )

    trial_count = 75
    if graph_construction == "gaussian":
        trial_count+=25
    if graph_weight_adj == "True":
        trial_count+=50
    study.optimize(objective, n_trials=trial_count)
    # Save Pareto plot
    fig = vis.plot_pareto_front(study, target_names=["Micro F1", "Macro F1"])
    #NOTE: Modify the name here
    fig.write_html(f"results/hyperparameter_plots/pareto_front_{label_type}_{graph_weight_adj}_{graph_construction}_{dataset}.html")

    # Save best hyperparameters
    os.makedirs('results/best_hyperparam', exist_ok=True)
    #NOTE: Modify the name here
    with open(f"results/best_hyperparams/{label_type}_{graph_weight_adj}_{graph_construction}_{dataset}_hyperparams.txt", "w") as f:
        for t in study.best_trials:
            f.write(f"Micro F1: {t.values[0]:.4f}, Macro F1: {t.values[1]:.4f}, Params: {t.params}\n")


    print("All datasets done!")
except Exception as e:
    print(e)
    raise
