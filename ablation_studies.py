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

def run_ablation_experiment(dataset, param_name, param_value, base_params):
    """Run a single experiment with specified parameter value using best hyperparameters for others"""
    command = (
        f"python comp_models.py "
        f"--dataset {dataset} "
        f"--graph_construction gaussian "
        f"--label_type pred_label "
        f"--lr {base_params['lr']} "
        f"--dropout_rate {base_params['dropout_rate']} "
        f"--sigma {base_params['sigma']} "
        f"--gamma {base_params['gamma']} "
        f"--graph_modification "
        f"--mode modularity "
        f"--device cuda "
        f"--ablation_study "
        f"--no-hyperparameter_tuning "
    )
    
    # Add the parameter being ablated with its current value
    if param_name == "mod_weight":
        command += f"--mod_weight {param_value} "
        command += f"--increase {base_params['increase']} --decrease {base_params['decrease']} --k {base_params['k']}"
    elif param_name == "k":
        command += f"--k {int(param_value)} "
        command += f"--increase {base_params['increase']} --decrease {base_params['decrease']} --mod_weight {base_params['mod_weight']}"
    elif param_name == "increase":
        command += f"--increase {param_value} "
        command += f"--k {base_params['k']} --decrease {base_params['decrease']} --mod_weight {base_params['mod_weight']}"
    elif param_name == "decrease":
        command += f"--decrease {param_value} "
        command += f"--k {base_params['k']} --increase {base_params['increase']} --mod_weight {base_params['mod_weight']}"
    
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
            # We expect at least the 6 main metrics
            if len(values) < 6:  
                print(f"Warning: Missing main metrics in result file for dataset {dataset}")
                return 0.0, 0.0, 0.0, 0.0
            
            try:
                # Extract all metrics
                val_micro, val_macro, test_micro, test_macro, true_mod, best_mod = map(float, values[:6])
                
                # The remaining values are per-class F1 scores
                per_class_f1 = [float(x.strip()) for x in values[6:] if x.strip()]
                
                return val_micro, val_macro, test_micro, test_macro, true_mod, best_mod, per_class_f1
            except ValueError as e:
                print(f"Error parsing values in file {result_file}: {str(e)}")
                print(f"Raw values: {values}")
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading results for dataset {dataset}: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0

def process_dataset_for_parameter(args):
    """Process a single dataset for a parameter ablation"""
    dataset, param_name, param_values = args
    print(f"\nPerforming {param_name} ablation for dataset: {dataset}")
    
    # Store results for this dataset
    results = {
        'param_value': [], 
        'val_micro_f1': [], 
        'val_macro_f1': [],
        'test_micro_f1': [],
        'test_macro_f1': [],
        'true_modularity': [],
        'best_modularity': []
    }
    
    # Initialize per-class F1 results (will be populated with column names later)
    per_class_results = {}
    
    # Load best hyperparameters
    try:
        base_params = load_hyperparams(dataset)
        print(f"Loaded best hyperparameters for {dataset}")
    except FileNotFoundError:
        print(f"No hyperparameters found for {dataset}, skipping...")
        return dataset, results
    
    ablation_dir = f'results/ablation_{param_name}'
    result_file = f'{ablation_dir}/ablation_{dataset}.csv'
    
    # Check if we already have results for this dataset
    if os.path.exists(result_file):
        print(f"Loading existing results for {dataset}...")
        df = pd.read_csv(result_file)
        # Load main metrics
        for col in df.columns:
            if col in results:
                results[col] = df[col].tolist()
            elif col.startswith('class_f1_'):
                per_class_results[col] = df[col].tolist()
    else:
        for value in param_values:
            print(f"Testing {param_name} = {value:.2f}")
            metrics = run_ablation_experiment(dataset, param_name, value, base_params)
            val_micro_f1, val_macro_f1, test_micro_f1, test_macro_f1, true_mod, best_mod, per_class_f1 = metrics
            
            # Store main metrics
            results['param_value'].append(value)
            results['val_micro_f1'].append(val_micro_f1)
            results['val_macro_f1'].append(val_macro_f1)
            results['test_micro_f1'].append(test_micro_f1)
            results['test_macro_f1'].append(test_macro_f1)
            results['true_modularity'].append(true_mod)
            results['best_modularity'].append(best_mod)
            
            # Store per-class F1 scores
            for i, f1 in enumerate(per_class_f1):
                col_name = f'class_f1_{i}'
                if col_name not in per_class_results:
                    per_class_results[col_name] = []
                per_class_results[col_name].append(f1)
            
            # Combine all results and save
            all_results = {**results, **per_class_results}
            df = pd.DataFrame(all_results)
            df.to_csv(result_file, index=False)
    
    # Create plots for this dataset
    plt.figure(figsize=(15, 10))
    
    # Plot Validation Metrics
    plt.subplot(2, 2, 1)
    plt.plot(results['param_value'], results['val_micro_f1'], 
            marker='o', label='Validation Micro F1')
    plt.plot(results['param_value'], results['val_macro_f1'], 
            marker='s', label='Validation Macro F1')
    plt.xlabel(f'{param_name}')
    plt.ylabel('F1 Score')
    plt.title(f'Validation Metrics - {dataset}')
    plt.grid(True)
    plt.legend()
    
    # Plot Test Metrics
    plt.subplot(2, 2, 2)
    plt.plot(results['param_value'], results['test_micro_f1'], 
            marker='o', label='Test Micro F1')
    plt.plot(results['param_value'], results['test_macro_f1'], 
            marker='s', label='Test Macro F1')
    plt.xlabel(f'{param_name}')
    plt.ylabel('F1 Score')
    plt.title(f'Test Metrics - {dataset}')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, 'plots', f'{dataset}_impact.png'))
    plt.close()
    
    return dataset, results

def analyze_parameter_ablation(param_name, param_values):
    """Run ablation study for a specific parameter"""
    datasets = ['mr', 'r8', 'r52', 'ohsumed', '20ng']
    
    # Store results
    results = {dataset: {
        'param_value': [], 
        'val_micro_f1': [], 
        'val_macro_f1': [],
        'test_micro_f1': [],
        'test_macro_f1': []
    } for dataset in datasets}
    
    # Create ablation results directory
    ablation_dir = f'results/ablation_{param_name}'
    os.makedirs(ablation_dir, exist_ok=True)
    os.makedirs(os.path.join(ablation_dir, 'plots'), exist_ok=True)
    
    # Prepare arguments for processing (sequential)
    args = [(dataset, param_name, param_values) for dataset in datasets]

    # Process datasets sequentially (one-by-one)
    print(f"Processing {len(datasets)} datasets sequentially")
    results_list = []
    for a in args:
        results_list.append(process_dataset_for_parameter(a))
    
    # Convert results list to dictionary
    results = {}
    for dataset, dataset_results in results_list:
        if dataset_results['param_value']:  # Only include datasets with results
            results[dataset] = dataset_results
    
    # Create comparison plots across all datasets
    fig = plt.figure(figsize=(20, 10))
    
    # Plot Validation Metrics
    plt.subplot(2, 2, 1)
    for dataset in datasets:
        if dataset in results and results[dataset]['param_value']:
            plt.plot(results[dataset]['param_value'], results[dataset]['val_micro_f1'], 
                    marker='o', label=f'{dataset} - Micro F1')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Validation Micro F1 Score')
    plt.title(f'Impact of {param_name} on Validation Micro F1')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for dataset in datasets:
        if dataset in results and results[dataset]['param_value']:
            plt.plot(results[dataset]['param_value'], results[dataset]['val_macro_f1'], 
                    marker='o', label=f'{dataset} - Macro F1')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Validation Macro F1 Score')
    plt.title(f'Impact of {param_name} on Validation Macro F1')
    plt.grid(True)
    plt.legend()
    
    # Plot Test Metrics
    plt.subplot(2, 2, 3)
    for dataset in datasets:
        if dataset in results and results[dataset]['param_value']:
            plt.plot(results[dataset]['param_value'], results[dataset]['test_micro_f1'], 
                    marker='o', label=f'{dataset} - Micro F1')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Test Micro F1 Score')
    plt.title(f'Impact of {param_name} on Test Micro F1')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for dataset in datasets:
        if dataset in results and results[dataset]['param_value']:
            plt.plot(results[dataset]['param_value'], results[dataset]['test_macro_f1'], 
                    marker='o', label=f'{dataset} - Macro F1')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Test Macro F1 Score')
    plt.title(f'Impact of {param_name} on Test Macro F1')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, 'plots', 'impact_all_datasets.png'))
    plt.close()

def run_parallel_ablation(param_config):
    """Helper function to run ablation study for a parameter in parallel"""
    param_name, param_values = param_config
    print(f"\nStarting ablation study for {param_name}")
    analyze_parameter_ablation(param_name, param_values)

if __name__ == "__main__":
    import multiprocessing as mp

    param_configs = {
        "mod_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
        # "increase": [1, 1.2, 1.4, 1.6, 1.8],
        # "k": [4, 6, 8, 10, 12, 14, 16, 18],
        # "decrease": [1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8]
    }

    for param_name, param_values in param_configs.items():
        run_parallel_ablation((param_name, param_values))
