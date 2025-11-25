import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from pathlib import Path
import ast

def load_hyperparams(dataset):
    """Load best hyperparameters from previous experiments"""
    filename = f"pred_label_True_gaussian_{dataset}_hyperparams.txt"
    filepath = os.path.join("results", "best_hyperparams", filename)
    
    with open(filepath, 'r') as f:
        line = f.readline()
        params_str = line.split('Params: ')[-1].strip()
        params = ast.literal_eval(params_str)
    return params

def run_gamma_experiment(dataset, gamma, params):
    """Run a single experiment with specified gamma value using best hyperparameters"""
    command = (
        f"python comp_models.py "
        f"--dataset {dataset} "
        f"--graph_construction gaussian "
        f"--label_type pred_label "
        f"--gamma {gamma} "
        f"--lr {params['lr']} "
        f"--dropout_rate {params['dropout_rate']} "
        f"--mod_weight {params['mod_weight']} "
        f"--sigma {params['sigma']} "
        f"--increase {params['increase']} "
        f"--decrease {params['decrease']} "
        f"--k {params['k']} "
        f"--graph_modification "
        f"--mode modularity "
        f"--device cuda "
        f"--ablation_study "  # New flag to handle ablation results separately
        f"--no-hyperparameter_tuning"
    )
    subprocess.run(command, shell=True)
    
    # Read the results
    result_file = f"ablation_result_{dataset}_gaussian_True_pred_label.txt"
    try:
        with open(result_file, 'r') as f:
            content = f.read().strip()
            if not content:  # Empty file
                print(f"Warning: Empty result file for dataset {dataset}")
                return 0.0, 0.0, 0.0, 0.0
            
            values = content.split(",")
            if len(values) != 4:  # Incorrect number of values
                print(f"Warning: Unexpected format in result file for dataset {dataset}")
                return 0.0, 0.0, 0.0, 0.0
            
            return tuple(map(float, values))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading results for dataset {dataset}: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0

def process_dataset(args):
    """Process a single dataset for gamma ablation"""
    dataset, gamma_values = args
    print(f"\nPerforming gamma ablation for dataset: {dataset}")
    
    # Store results for this dataset
    results = {
        'gamma': [], 
        'val_micro_f1': [], 
        'val_macro_f1': [],
        'test_micro_f1': [],
        'test_macro_f1': []
    }
    
    # Load best hyperparameters
    try:
        params = load_hyperparams(dataset)
        print(f"Loaded best hyperparameters for {dataset}")
    except FileNotFoundError:
        print(f"No hyperparameters found for {dataset}, skipping...")
        return dataset, results
    
    result_file = f'results/ablation_gamma/gamma_ablation_{dataset}.csv'
    
    # Check if we already have results for this dataset
    if os.path.exists(result_file):
        print(f"Loading existing results for {dataset}...")
        df = pd.read_csv(result_file)
        results['gamma'] = df['gamma'].tolist()
        results['val_micro_f1'] = df['val_micro_f1'].tolist()
        results['val_macro_f1'] = df['val_macro_f1'].tolist()
        results['test_micro_f1'] = df['test_micro_f1'].tolist()
        results['test_macro_f1'] = df['test_macro_f1'].tolist()
    else:
        for gamma in gamma_values:
            print(f"Testing gamma = {gamma:.2f}")
            val_micro_f1, val_macro_f1, test_micro_f1, test_macro_f1 = run_gamma_experiment(dataset, gamma, params)
            
            results['gamma'].append(gamma)
            results['val_micro_f1'].append(val_micro_f1)
            results['val_macro_f1'].append(val_macro_f1)
            results['test_micro_f1'].append(test_micro_f1)
            results['test_macro_f1'].append(test_macro_f1)
            
            # Save intermediate results
            df = pd.DataFrame(results)
            df.to_csv(result_file, index=False)
    
    # Create plots for this dataset
    plt.figure(figsize=(15, 10))
    
    # Plot Validation Metrics
    plt.subplot(2, 2, 1)
    plt.plot(results['gamma'], results['val_micro_f1'], 
            marker='o', label='Validation Micro F1')
    plt.plot(results['gamma'], results['val_macro_f1'], 
            marker='s', label='Validation Macro F1')
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('F1 Score')
    plt.title(f'Validation Metrics - {dataset}')
    plt.grid(True)
    plt.legend()
    
    # Plot Test Metrics
    plt.subplot(2, 2, 2)
    plt.plot(results['gamma'], results['test_micro_f1'], 
            marker='o', label='Test Micro F1')
    plt.plot(results['gamma'], results['test_macro_f1'], 
            marker='s', label='Test Macro F1')
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('F1 Score')
    plt.title(f'Test Metrics - {dataset}')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('results/ablation_gamma/plots', f'{dataset}_impact.png'))
    plt.close()
    
    return dataset, results

def analyze_gamma_ablation():
    # Datasets to analyze
    datasets = ['r8', 'r52', 'ohsumed']
    
    # Range of gamma values to test
    gamma_values = np.linspace(0.1, 20.0, 40)
    
    # Create ablation results directory
    os.makedirs('results/ablation_gamma', exist_ok=True)
    os.makedirs('results/ablation_gamma/plots', exist_ok=True)
    
    # Prepare arguments for parallel processing
    args = [(dataset, gamma_values) for dataset in datasets]
    
    # Process datasets in parallel
    import multiprocessing as mp
    num_processes = min(len(datasets), mp.cpu_count())
    print(f"Processing {len(datasets)} datasets using {num_processes} processes")
    
    with mp.Pool(processes=num_processes) as pool:
        results_list = pool.map(process_dataset, args)
    
    # Convert results list to dictionary
    results = {}
    for dataset, dataset_results in results_list:
        if dataset_results['gamma']:  # Only include datasets with results
            results[dataset] = dataset_results
    
    # Create comparison plots across all datasets
    fig = plt.figure(figsize=(20, 10))
    
    # Plot Validation Metrics
    plt.subplot(2, 2, 1)
    for dataset in datasets:
        if dataset in results and results[dataset]['gamma']:
            plt.plot(results[dataset]['gamma'], results[dataset]['val_micro_f1'], 
                    marker='o', label=f'{dataset} - Micro F1')
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Validation Micro F1 Score')
    plt.title('Impact of γ on Validation Micro F1')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for dataset in datasets:
        if dataset in results and results[dataset]['gamma']:
            plt.plot(results[dataset]['gamma'], results[dataset]['val_macro_f1'], 
                    marker='o', label=f'{dataset} - Macro F1')
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Validation Macro F1 Score')
    plt.title('Impact of γ on Validation Macro F1')
    plt.grid(True)
    plt.legend()
    
    # Plot Test Metrics
    plt.subplot(2, 2, 3)
    for dataset in datasets:
        if dataset in results and results[dataset]['gamma']:
            plt.plot(results[dataset]['gamma'], results[dataset]['test_micro_f1'], 
                    marker='o', label=f'{dataset} - Micro F1')
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Test Micro F1 Score')
    plt.title('Impact of γ on Test Micro F1')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for dataset in datasets:
        if dataset in results and results[dataset]['gamma']:
            plt.plot(results[dataset]['gamma'], results[dataset]['test_macro_f1'], 
                    marker='o', label=f'{dataset} - Macro F1')
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Test Macro F1 Score')
    plt.title('Impact of γ on Test Macro F1')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('results/ablation_gamma/plots', 'gamma_impact_all_datasets.png'))
    plt.close()

if __name__ == "__main__":
    analyze_gamma_ablation()
