import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc

def generate_advanced_visualizations(results_file_path, output_prefix="analysis"):
    """
    Generate advanced visualizations for analyzing precision-recall tradeoffs
    
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
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # 1. PR curves for each k value across chunk sizes
    plt.figure(figsize=(14, 10))
    
    for k in k_values:
        k_key = f"k={k}"
        recalls = []
        precisions = []
        chunk_labels = []
        
        for chunk_size in chunk_sizes:
            chunk_key = f"size={chunk_size}"
            metrics = results['by_chunk_size'][chunk_key]['by_k'][k_key]
            
            recalls.append(metrics['avg_recall'])
            precisions.append(metrics['avg_precision'])
            chunk_labels.append(chunk_size)
        
        # Sort by recall for proper curve
        sorted_indices = np.argsort(recalls)
        sorted_recalls = [recalls[i] for i in sorted_indices]
        sorted_precisions = [precisions[i] for i in sorted_indices]
        sorted_chunk_labels = [chunk_labels[i] for i in sorted_indices]
        
        # Calculate AUC
        try:
            pr_auc = auc(sorted_recalls, sorted_precisions)
            auc_label = f", AUC={pr_auc:.4f}"
        except:
            auc_label = ""
        
        # Plot PR curve
        plt.plot(sorted_recalls, sorted_precisions, marker='o', label=f"k={k}{auc_label}", linewidth=2)
        
        # Add chunk size annotations
        for i, chunk_size in enumerate(sorted_chunk_labels):
            plt.annotate(f"{chunk_size}", 
                        (sorted_recalls[i], sorted_precisions[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
    
    plt.title("Precision-Recall Curves for Different k Values Across Chunk Sizes")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pr_curves_by_k.png", dpi=300)
    plt.close()
    
    # 2. Recall vs Total Retrieved Text (k × chunk_size)
    plt.figure(figsize=(14, 10))
    
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    
    data_points = []
    
    for chunk_size in chunk_sizes:
        chunk_key = f"size={chunk_size}"
        
        for k_idx, k in enumerate(k_values):
            k_key = f"k={k}"
            
            metrics = results['by_chunk_size'][chunk_key]['by_k'][k_key]
            avg_recall = metrics['avg_recall']
            total_retrieved_text = k * chunk_size
            
            data_points.append({
                'chunk_size': chunk_size,
                'k': k,
                'total_retrieved_text': total_retrieved_text,
                'avg_recall': avg_recall
            })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data_points)
    
    # Sort by total retrieved text
    df = df.sort_values('total_retrieved_text')
    
    # Plot with color representing k value
    scatter = plt.scatter(
        df['total_retrieved_text'], 
        df['avg_recall'], 
        c=df['k'], 
        cmap='viridis', 
        s=100, 
        alpha=0.7,
        edgecolor='k'
    )
    
    # Add annotations
    for i, row in df.iterrows():
        plt.annotate(f"{row['chunk_size']}", 
                    (row['total_retrieved_text'], row['avg_recall']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    # Add colorbar for k values
    cbar = plt.colorbar(scatter)
    cbar.set_label('k value')
    
    # Add trend line
    z = np.polyfit(df['total_retrieved_text'], df['avg_recall'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(df['total_retrieved_text']), p(sorted(df['total_retrieved_text'])), 
             "r--", alpha=0.7, label=f"Trend line (y={z[0]:.6f}x + {z[1]:.4f})")
    
    plt.title("Recall vs Total Retrieved Text (k × chunk_size)")
    plt.xlabel("Total Retrieved Text (tokens)")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_recall_vs_retrieved_text.png", dpi=300)
    plt.close()
    
    # 3. Precision-per-token and Recall-per-token
    plt.figure(figsize=(14, 10))
    
    efficiency_data = []
    
    for chunk_size in chunk_sizes:
        chunk_key = f"size={chunk_size}"
        
        for k in k_values:
            k_key = f"k={k}"
            
            metrics = results['by_chunk_size'][chunk_key]['by_k'][k_key]
            total_tokens = k * chunk_size
            
            precision_per_token = metrics['avg_precision'] / total_tokens * 1000  # Multiply by 1000 for readability
            recall_per_token = metrics['avg_recall'] / total_tokens * 1000  # Multiply by 1000 for readability
            
            efficiency_data.append({
                'chunk_size': chunk_size,
                'k': k,
                'total_tokens': total_tokens,
                'precision_per_token': precision_per_token,
                'recall_per_token': recall_per_token,
                'config': f"C{chunk_size}-K{k}"
            })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Sort by efficiency metrics
    efficiency_df = efficiency_df.sort_values('precision_per_token', ascending=False)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        efficiency_df['precision_per_token'],
        efficiency_df['recall_per_token'],
        c=efficiency_df['total_tokens'],
        s=100,
        alpha=0.7,
        cmap='viridis',
        norm=plt.Normalize(min(efficiency_df['total_tokens']), max(efficiency_df['total_tokens']))
    )
    
    # Add config labels
    for i, row in efficiency_df.iterrows():
        plt.annotate(
            f"C{row['chunk_size']}-K{row['k']}", 
            (row['precision_per_token'], row['recall_per_token']),
            textcoords="offset points",
            xytext=(5, 0),
            ha='left'
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Retrieved Tokens')
    
    plt.title("Token Efficiency: Precision-per-token vs Recall-per-token (×1000)")
    plt.xlabel("Precision per 1000 tokens")
    plt.ylabel("Recall per 1000 tokens")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_token_efficiency.png", dpi=300)
    plt.close()
    
    # Calculate and print the top 5 most efficient configurations
    print("Top 5 Configurations by Precision per Token:")
    print(efficiency_df[['config', 'precision_per_token']].head(5))
    
    print("\nTop 5 Configurations by Recall per Token:")
    print(efficiency_df.sort_values('recall_per_token', ascending=False)[['config', 'recall_per_token']].head(5))
    
    # Create a combined efficiency score (harmonic mean of precision-per-token and recall-per-token)
    efficiency_df['combined_efficiency'] = 2 * (efficiency_df['precision_per_token'] * efficiency_df['recall_per_token']) / (efficiency_df['precision_per_token'] + efficiency_df['recall_per_token'])
    
    print("\nTop 5 Configurations by Combined Efficiency (Harmonic Mean):")
    print(efficiency_df.sort_values('combined_efficiency', ascending=False)[['config', 'combined_efficiency']].head(5))

    # Save the efficiency data as CSV for further analysis
    efficiency_df.to_csv(f"{output_prefix}_efficiency_metrics.csv", index=False)
    print(f"\nEfficiency metrics saved to: {output_prefix}_efficiency_metrics.csv")
    
    # 4. AUC calculation for fixed chunk sizes across k values
    print("\nCalculating AUC for fixed chunk sizes across k values...")
    plt.figure(figsize=(14, 10))
    
    # Create dataframe to store AUC results
    auc_results = []
    
    # Process each chunk size
    for chunk_size in chunk_sizes:
        chunk_key = f"size={chunk_size}"
        
        recalls_by_k = []
        precisions_by_k = []
        k_labels = []
        
        # Collect data for all k values for this chunk size
        for k in k_values:
            k_key = f"k={k}"
            
            try:
                metrics = results['by_chunk_size'][chunk_key]['by_k'][k_key]
                recalls_by_k.append(metrics['avg_recall'])
                precisions_by_k.append(metrics['avg_precision'])
                k_labels.append(k)
            except KeyError:
                print(f"Warning: Data not found for chunk_size={chunk_size}, k={k}")
                continue
        
        if len(recalls_by_k) < 2:
            print(f"Not enough data points for chunk_size={chunk_size}, skipping AUC calculation")
            continue
            
        # Sort by recall for proper curve calculation
        sort_indices = np.argsort(recalls_by_k)
        sorted_recalls = [recalls_by_k[i] for i in sort_indices]
        sorted_precisions = [precisions_by_k[i] for i in sort_indices]
        sorted_k_labels = [k_labels[i] for i in sort_indices]
        
        # Calculate AUC for this chunk size
        try:
            chunk_auc = auc(sorted_recalls, sorted_precisions)
            auc_results.append({
                'chunk_size': chunk_size,
                'auc': chunk_auc,
                'num_k_points': len(sorted_recalls)
            })
            
            # Plot PR curve for this chunk size
            plt.plot(sorted_recalls, sorted_precisions, marker='o', 
                     label=f"Chunk Size={chunk_size}, AUC={chunk_auc:.4f}", linewidth=2)
            
            # Add k value annotations to each point
            for i, k in enumerate(sorted_k_labels):
                plt.annotate(f"k={k}", 
                            (sorted_recalls[i], sorted_precisions[i]),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center')
        except Exception as e:
            print(f"AUC calculation failed for chunk_size={chunk_size}: {str(e)}")
    
    plt.title("Precision-Recall Curves for Fixed Chunk Sizes Across k Values")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pr_curves_by_chunk_size.png", dpi=300)
    plt.close()
    
    # Create DataFrame and save AUC results
    auc_df = pd.DataFrame(auc_results)
    if not auc_df.empty:
        auc_df = auc_df.sort_values('auc', ascending=False)
        
        print("\nAUC for Fixed Chunk Sizes Across k Values:")
        print(auc_df)
        
        auc_df.to_csv(f"{output_prefix}_auc_by_chunk_size.csv", index=False)
        print(f"AUC metrics saved to: {output_prefix}_auc_by_chunk_size.csv")
        
        # Create bar chart of AUC values by chunk size
        plt.figure(figsize=(10, 6))
        bars = plt.bar(auc_df['chunk_size'].astype(str), auc_df['auc'], color='skyblue')
        
        # Add AUC values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title("AUC by Chunk Size (Across k Values)")
        plt.xlabel("Chunk Size")
        plt.ylabel("AUC")
        plt.ylim(0, max(auc_df['auc']) * 1.1)  # Add some padding above the highest bar
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_auc_by_chunk_size_bar.png", dpi=300)
        plt.close()
    else:
        print("No AUC results to display")

    # Calculate AUC for each chunk size and k combination
    detailed_auc_results = []
    
    for chunk_size in chunk_sizes:
        chunk_key = f"size={chunk_size}"
        
        for k_start_idx in range(len(k_values)):
            # Calculate AUC for each k range (k_start to max_k)
            k_start = k_values[k_start_idx]
            
            recalls = []
            precisions = []
            k_vals_used = []
            
            for k in k_values[k_start_idx:]:
                k_key = f"k={k}"
                
                try:
                    metrics = results['by_chunk_size'][chunk_key]['by_k'][k_key]
                    recalls.append(metrics['avg_recall'])
                    precisions.append(metrics['avg_precision'])
                    k_vals_used.append(k)
                except KeyError:
                    continue
            
            if len(recalls) < 2:
                continue
                
            # Sort by recall
            sort_indices = np.argsort(recalls)
            sorted_recalls = [recalls[i] for i in sort_indices]
            sorted_precisions = [precisions[i] for i in sort_indices]
            
            try:
                pr_auc = auc(sorted_recalls, sorted_precisions)
                detailed_auc_results.append({
                    'chunk_size': chunk_size,
                    'k_start': k_start,
                    'k_range': f"{k_start}-{max(k_vals_used)}",
                    'num_points': len(sorted_recalls),
                    'min_recall': min(sorted_recalls),
                    'max_recall': max(sorted_recalls),
                    'auc': pr_auc
                })
            except Exception as e:
                print(f"Detailed AUC calculation failed for chunk_size={chunk_size}, k_start={k_start}: {str(e)}")
    
    # Create and save detailed AUC results
    if detailed_auc_results:
        detailed_df = pd.DataFrame(detailed_auc_results)
        detailed_df = detailed_df.sort_values(['chunk_size', 'k_start'])
        detailed_df.to_csv(f"{output_prefix}_detailed_auc_by_k_range.csv", index=False)
        print(f"Detailed AUC metrics saved to: {output_prefix}_detailed_auc_by_k_range.csv")

# Example usage
if __name__ == "__main__":
    results_file = "evaluation_results_whole_corpus.json"
    generate_advanced_visualizations(results_file, output_prefix="advanced_analysis")