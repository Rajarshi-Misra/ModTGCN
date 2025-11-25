import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_gamma_results():
    # Datasets to analyze
    datasets = ['mr', 'r8', 'r52', 'ohsumed', '20ng']
    
    plt.figure(figsize=(12, 6))
    
    # Plot Micro F1
    plt.subplot(1, 2, 1)
    for dataset in datasets:
        # Read results
        df = pd.read_csv(f'results/gamma_analysis/gamma_impact_{dataset}.csv')
        plt.plot(df['gamma'], df['micro_f1'], marker='o', label=dataset)
    
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Micro F1 Score')
    plt.title('Impact of γ on Micro F1 Score')
    plt.grid(True)
    plt.legend()
    
    # Plot Macro F1
    plt.subplot(1, 2, 2)
    for dataset in datasets:
        # Read results
        df = pd.read_csv(f'results/gamma_analysis/gamma_impact_{dataset}.csv')
        plt.plot(df['gamma'], df['macro_f1'], marker='o', label=dataset)
    
    plt.xlabel('γ (Resolution Parameter)')
    plt.ylabel('Macro F1 Score')
    plt.title('Impact of γ on Macro F1 Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/gamma_analysis/gamma_impact_all_datasets.png')
    plt.close()

if __name__ == "__main__":
    plot_gamma_results()
