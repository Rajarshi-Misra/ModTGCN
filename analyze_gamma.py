import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from pathlib import Path

def run_experiment(dataset, gamma):
    # Parameters that stay constant
    graph_construction = "cosine"
    graph_weight_adj = "False"
    label_type = "pred_label"
    lr = 0.01  # Common learning rate
    dropout = 0.3  # Common dropout rate
    mod_weight = 0.5  # Common modularity weight
    
    # Run the model directly with fixed parameters
    command = (
        f"python comp_models.py "
        f"--dataset {dataset} "
        f"--graph_construction {graph_construction} "
        f"--label_type {label_type} "
        f"--gamma {gamma} "
        f"--lr {lr} "
        f"--dropout_rate {dropout} "
        f"--mod_weight {mod_weight} "
        f"--no-hyperparameter_tuning "  # Disable hyperparameter tuning
        f"--mode modularity "
        f"--device cuda"
    )
    
    if graph_weight_adj == "True":
        command += " --graph_modification"
        
    subprocess.run(command, shell=True)
    
    # Read the results
    result_file = f"result_{dataset}_{graph_construction}_{graph_weight_adj}_{label_type}.txt"
    with open(result_file, 'r') as f:
        content = f.read().strip()
        val_micro_f1, val_macro_f1 = map(float, content.split(","))
    
    return val_micro_f1, val_macro_f1

def analyze_gamma_impact():
    # Datasets to analyze
    datasets = ['mr', 'r8', 'r52', 'ohsumed',]
    
    # Range of gamma values to test
    gamma_values = np.linspace(0.1, 20.0, 40)  # 20 points between 0.1 and 10
    
    # Store results
    results = {dataset: {'gamma': [], 'micro_f1': [], 'macro_f1': []} for dataset in datasets}
    
    # Ensure results directory exists
    os.makedirs('results/gamma_analysis', exist_ok=True)
    
    # Process each dataset separately to manage memory
    for dataset in datasets:
        result_file = f'results/gamma_analysis/gamma_impact_{dataset}.csv'
        
        # Check if we already have results for this dataset
        if os.path.exists(result_file):
            print(f"Results already exist for {dataset}, loading from file...")
            df = pd.read_csv(result_file)
            results[dataset]['gamma'] = df['gamma'].tolist()
            results[dataset]['micro_f1'] = df['micro_f1'].tolist()
            results[dataset]['macro_f1'] = df['macro_f1'].tolist()
            continue
    
    # Run experiments
    for dataset in datasets:
        print(f"\nAnalyzing dataset: {dataset}")
        for gamma in gamma_values:
            print(f"Testing gamma = {gamma:.2f}")
            micro_f1, macro_f1 = run_experiment(dataset, gamma)
            
            results[dataset]['gamma'].append(gamma)
            results[dataset]['micro_f1'].append(micro_f1)
            results[dataset]['macro_f1'].append(macro_f1)
            
        # Save intermediate results
        df = pd.DataFrame({
            'gamma': results[dataset]['gamma'],
            'micro_f1': results[dataset]['micro_f1'],
            'macro_f1': results[dataset]['macro_f1']
        })
        os.makedirs('results/gamma_analysis', exist_ok=True)
        df.to_csv(f'results/gamma_analysis/gamma_impact_{dataset}.csv', index=False)
    
    # Create plots
    plt.figure(figsize=(12, 6))
    
    # Plot Micro F1
    plt.subplot(1, 2, 1)
    for dataset in datasets:
        plt.plot(results[dataset]['gamma'], results[dataset]['micro_f1'], 
                marker='o', label=dataset)
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Micro F1 Score')
    plt.title('Impact of γ on Micro F1 Score')
    plt.grid(True)
    plt.legend()
    
    # Plot Macro F1
    plt.subplot(1, 2, 2)
    for dataset in datasets:
        plt.plot(results[dataset]['gamma'], results[dataset]['macro_f1'], 
                marker='o', label=dataset)
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Macro F1 Score')
    plt.title('Impact of γ on Macro F1 Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/gamma_analysis/gamma_impact_all_datasets.png')
    plt.close()

if __name__ == "__main__":
    analyze_gamma_impact()
