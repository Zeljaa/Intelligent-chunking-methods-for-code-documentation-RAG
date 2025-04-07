import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_results(results_file_path, output_prefix="plot"):
    """
    Create visualizations from evaluation results
    
    :param results_file_path: Path to the evaluation results JSON file
    :param output_prefix: Prefix for output filenames
    """
    # Load results
    with open(results_file_path, 'r') as f:
        results = json.load(f)
    
    # Extract chunk sizes and k values
    chunk_sizes = [int(size.split('=')[1]) for size in results['by_chunk_size'].keys()]
    chunk_sizes.sort()
    
    k_values = []
    for chunk_size in results['by_chunk_size'].keys():
        for k in results['by_chunk_size'][chunk_size]['by_k'].keys():
            k_val = int(k.split('=')[1])
            if k_val not in k_values:
                k_values.append(k_val)
    k_values.sort()
    
    # Create data structure for plots
    metrics = ['avg_precision', 'avg_recall', 'avg_f1']
    metric_names = ['Precision', 'Recall', 'F1 Score']
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_palette("viridis")
    
    # 1. Heatmaps for each metric
    for metric, metric_name in zip(metrics, metric_names):
        plt.figure(figsize=(10, 8))
        
        # Create data for heatmap
        data = np.zeros((len(chunk_sizes), len(k_values)))
        
        for i, chunk_size in enumerate(chunk_sizes):
            chunk_key = f"size={chunk_size}"
            
            for j, k in enumerate(k_values):
                k_key = f"k={k}"
                
                # Get metric value
                value = results['by_chunk_size'][chunk_key]['by_k'][k_key][metric]
                data[i, j] = value
        
        # Create heatmap
        heatmap = sns.heatmap(
            data, 
            annot=True, 
            fmt=".4f",
            cmap="viridis",
            xticklabels=k_values,
            yticklabels=chunk_sizes
        )
        
        plt.title(f"{metric_name} by Chunk Size and k")
        plt.xlabel("k value")
        plt.ylabel("Chunk Size")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_prefix}_{metric}_heatmap.png", dpi=300)
        plt.close()
    
    # 2. Line plots for each k value showing metrics across chunk sizes
    for k in k_values:
        plt.figure(figsize=(12, 8))
        
        k_key = f"k={k}"
        data = {metric: [] for metric in metrics}
        error_data = {metric: [] for metric in metrics}
        
        for chunk_size in chunk_sizes:
            chunk_key = f"size={chunk_size}"
            
            for metric in metrics:
                value = results['by_chunk_size'][chunk_key]['by_k'][k_key][metric]
                std_metric = f"std_{metric.split('_')[1]}"  # e.g., avg_precision -> std_precision
                std_value = results['by_chunk_size'][chunk_key]['by_k'][k_key][std_metric]
                
                data[metric].append(value)
                error_data[metric].append(std_value)
        
        # Plot lines with error bands
        for metric, metric_name, color in zip(metrics, metric_names, ['blue', 'green', 'red']):
            plt.errorbar(
                chunk_sizes, 
                data[metric], 
                yerr=error_data[metric], 
                label=metric_name,
                capsize=5,
                color=color,
                marker='o'
            )
        
        plt.title(f"Performance Metrics by Chunk Size (k={k})")
        plt.xlabel("Chunk Size")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_prefix}_k{k}_line_plot.png", dpi=300)
        plt.close()
    
    # 3. Line plots for each chunk size showing metrics across k values
    for i, chunk_size in enumerate(chunk_sizes):
        plt.figure(figsize=(12, 8))
        
        chunk_key = f"size={chunk_size}"
        data = {metric: [] for metric in metrics}
        error_data = {metric: [] for metric in metrics}
        
        for k in k_values:
            k_key = f"k={k}"
            
            for metric in metrics:
                value = results['by_chunk_size'][chunk_key]['by_k'][k_key][metric]
                std_metric = f"std_{metric.split('_')[1]}"
                std_value = results['by_chunk_size'][chunk_key]['by_k'][k_key][std_metric]
                
                data[metric].append(value)
                error_data[metric].append(std_value)
        
        # Plot lines with error bands
        for metric, metric_name, color in zip(metrics, metric_names, ['blue', 'green', 'red']):
            plt.errorbar(
                k_values, 
                data[metric], 
                yerr=error_data[metric], 
                label=metric_name,
                capsize=5,
                color=color,
                marker='o'
            )
        
        plt.title(f"Performance Metrics by k (Chunk Size={chunk_size})")
        plt.xlabel("k value")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_prefix}_chunk{chunk_size}_line_plot.png", dpi=300)
        plt.close()
    
    print(f"Visualizations saved with prefix: {output_prefix}")

# Example usage
if __name__ == "__main__":
    results_file = "evaluation_results_whole.json"
    visualize_results(results_file, output_prefix="eval_whole_result")