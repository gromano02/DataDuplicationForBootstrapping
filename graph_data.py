import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def read_and_prepare_data(csv_file):
    """
    Reads the data from a CSV file and structures it for the plot_mean function.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        dict: A dictionary with factors as keys and metrics as values for the given method.
        list: List of unique factors for plotting.
    """
    # Read the CSV into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Get unique factors and metrics
    factors = sorted(data['Factor'].unique())
    method_name = data[' Method'].iloc[0]  # Assuming the same method for all rows

    # Create a dictionary to store metrics by factor
    metrics = {factor: {' F-score': [], ' Precision': [], ' Recall': []} for factor in factors}
    for _, row in data.iterrows():
        metrics[row['Factor']][' F-score'].append(row[' F-score'])
        metrics[row['Factor']][' Precision'].append(row[' Precision'])
        metrics[row['Factor']][' Recall'].append(row[' Recall'])

    return metrics, factors, method_name

def read_and_prepare_data2(csv_file):
    """
    Reads the data from a CSV file and structures it for the plot_mean function.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        dict: A dictionary with factors as keys and metrics as values for the given method.
        list: List of unique factors for plotting.
    """
    # Read the CSV into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Get unique factors and metrics
    factors = sorted(data['Factor'].unique())
    method_name = data['Method'].iloc[0]  # Assuming the same method for all rows

    # Create a dictionary to store metrics by factor
    metrics = {factor: {'F1-Score': [], 'Correlation': [], 'Strength': []} for factor in factors}
    for _, row in data.iterrows():
        metrics[row['Factor']]['F1-Score'].append(row['F1-Score'])
        metrics[row['Factor']]['Correlation'].append(row['Correlation'])
        metrics[row['Factor']]['Strength'].append(row['Strength'])

    return metrics, factors, method_name

def plot_mean(database, metrics, metric_name, method_name):
    """
    Generates a boxplot with overlaid mean scatter plot for a given metric.
    Args:
        database (str): Name of the database or dataset.
        metrics (dict): Dictionary containing metric values for each factor.
        metric_name (str): The metric to plot (e.g., 'F-score').
        method_name (str): Name of the method (e.g., 'standard').
    """
    plt.figure(figsize=(10, 8))

    # Prepare the data for plotting a boxplot and scatter plot
    boxplot_data = []
    mean_scores = []
    factors = sorted(metrics.keys())

    for factor in factors:
        scores = metrics[factor][metric_name]
        # Remove NaN values for plotting
        scores = [score for score in scores if not np.isnan(score)]
        boxplot_data.append(scores)
        mean_scores.append(np.mean(scores))  # Calculate the mean for each factor

    # Create a boxplot
    plt.boxplot(boxplot_data, positions=np.arange(len(factors)), widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                whiskerprops=dict(color='black'),
                flierprops=dict(marker='o', color='red', markersize=5),
                medianprops=dict(color='blue', linewidth=2))

    # Overlay a scatter plot of the mean scores (adjusting positions slightly for visibility)
    plt.scatter(np.arange(len(factors)), mean_scores, marker='o', color='black', label='Mean Score', s=100, zorder=5)
    plt.title(f'{database.capitalize()} {method_name.capitalize()}{metric_name} Distribution', fontsize=25, pad=20)
    plt.xlabel('Duplication Factor', fontsize=20)
    plt.ylabel(f'{metric_name}', fontsize=20)
    plt.xticks(np.arange(len(factors)), factors, fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, 1)  # Set the y-axis scale explicitly from 0 to 1
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)  # Slightly transparent grid for better visibility
    plt.tight_layout()
    plt.savefig(f'Figures/{database}_{method_name}_{metric_name.lower()}_distribution.jpeg', format='jpeg', dpi=300)  # Save the plot
    plt.show()  # Display the plot


def plot_mean2(database, metrics, metric_name, method_name):
    """
    Generates a boxplot with overlaid mean scatter plot for a given metric.
    Args:
        database (str): Name of the database or dataset.
        metrics (dict): Dictionary containing metric values for each factor.
        metric_name (str): The metric to plot (e.g., 'F-score').
        method_name (str): Name of the method (e.g., 'standard').
    """
    plt.figure(figsize=(10, 8))

    # Prepare the data for plotting a boxplot and scatter plot
    boxplot_data = []
    mean_scores = []
    factors = sorted(metrics.keys())

    for factor in factors:
        scores = metrics[factor][metric_name]
        # Remove NaN values for plotting
        scores = [score for score in scores if not np.isnan(score)]
        boxplot_data.append(scores)
        mean_scores.append(np.mean(scores))  # Calculate the mean for each factor

    # Create a boxplot
    plt.boxplot(boxplot_data, positions=np.arange(len(factors)), widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                whiskerprops=dict(color='black'),
                flierprops=dict(marker='o', color='red', markersize=5),
                medianprops=dict(color='blue', linewidth=2))

    # Overlay a scatter plot of the mean scores (adjusting positions slightly for visibility)
    plt.scatter(np.arange(len(factors)), mean_scores, marker='o', color='black', label='Mean Score', s=100, zorder=5)
    plt.title(f'{database.capitalize()} {method_name.capitalize()} Decision Tree {metric_name.capitalize()} Distribution', fontsize=25, pad=20)
    plt.xlabel('Duplication Factor', fontsize=20)
    plt.ylabel(f'{metric_name}', fontsize=20)
    plt.xticks(np.arange(len(factors)), factors, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)  # Slightly transparent grid for better visibility
    plt.tight_layout()
    plt.savefig(f'Figures/{database}_{method_name}_decision_tree_{metric_name.lower()}_distribution.jpeg', format='jpeg', dpi=300)  # Save the plot
    plt.show()  # Display the plot

# Example usage
if __name__ == "__main__":
    os.makedirs("Figures", exist_ok=True)
    databases = ['oil', 'wine', 'spam', 'mammography', 'breast_cancer', 'banknote']
    methods = ['undersampling', 'standard', 'class_weighted']

    for database in databases:
        for method in methods:
            csv_file = f"Data/{database}_{method}.csv"  # Format the CSV filename

            try:
                # Read and prepare the data
                metrics, factors, method_name = read_and_prepare_data(csv_file)

                # Plot for each metric
                for metric_name in [' F-score', ' Precision', ' Recall']:
                    plot_mean(database, metrics, metric_name, method_name)
            except FileNotFoundError:
                print(f"File not found: {csv_file}. Skipping.")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

            csv_file = f"Data/{database}_decision_trees_{method}.csv"  # Format the CSV filename

            try:
                # Read and prepare the data
                metrics, factors, method_name = read_and_prepare_data2(csv_file)

                # Plot for each metric
                for metric_name in ['F1-Score', 'Correlation', 'Strength']:
                    plot_mean2(database, metrics, metric_name, method_name)
            except FileNotFoundError:
                print(f"File not found: {csv_file}. Skipping.")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
