import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def histogram_classes(positive, negative, metric, ax, **kwargs):
    """
    Plot a histogram comparing the positive and negative classes.
    Args:
        positive (list or array-like): data for the positive class
        negative (list or array-like): data for the negative class
        metric (str): The name of the metric being plotted
        ax (matplotlib.axes.Axes): matplotlib axes object
        kwargs (dict): Additional keyword arguments to pass to seaborn's histplot function.
     """
    # Plot the histogram for positive and negative degrees
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 5))
    sns.histplot(positive, color="skyblue", label="Positive", ax=ax, **kwargs)
    sns.histplot(negative, color="salmon", label="Negative", ax=ax, **kwargs)
    ax.set_title(f"Average {metric}")
    ax.set_xlabel(f"Average {metric}")
    ax.set_ylabel("Frequency")
    ax.grid(linestyle=':')
    ax.legend()


def feature_bar_plots(x_pos, x_neg, save_path=None):
    """
    Plot bar plot of the mean and standard deviation for a feature
    Args:
        x_pos (np.array): array of feature values for positive class
        x_neg (np.array): array of feature values for negative class
        save_path (str): Path to save the bar plot image
    """
    # Compute mean and standard deviation for each class
    x_pos_mean = x_pos.mean(axis=0)
    x_pos_std = x_pos.std(axis=0).squeeze()
    
    x_neg_mean = x_neg.mean(axis=0)
    x_neg_std = x_neg.std(axis=0).squeeze()
    
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(4, 4))
    plt.bar(range(1, x_pos_mean.shape[0]+1), x_pos_mean, color='skyblue', alpha=0.8, label='Positive', 
            yerr=x_pos_std, error_kw={'alpha': 0.3, 'capsize': 5, 'ecolor': 'b'})
    plt.bar(range(1, x_neg_mean.shape[0]+1), x_neg_mean, color='salmon', alpha=0.8, label='Negative',
            yerr=x_neg_std, error_kw={'alpha': 0.3, 'capsize': 5, 'ecolor': 'r'})
    plt.axhline(y=0, color='b', linestyle='--')
    plt.xticks(range(1, x_pos_mean.shape[0]+1))
    plt.grid(linestyle=':')
    plt.title('Mean and Standard Deviation of Features per Node')
    plt.xlabel('Node ID')
    plt.ylabel('Mean Value')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    
def violin_plots(data_, save_path=None):
    """
    Plot violin plot of the features
    Args:
        data_ (pd.DataFrame): DataFrame containing the class, feature name and value as columns
        save_path (str): Path to save the image
    """
    # Plot the violin plot of the features                                                                                                                                                                                                                                                                                                                                                                            
    sns.set(style="whitegrid")
    palette = {1: 'skyblue', 0: 'salmon'}
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x='Protein', y='Expression', hue='class', data=data_, split=True, inner="quart", palette=palette)
    ax.legend(handles=ax.legend_.legendHandles, labels=['Negative', 'Positive'])
    plt.axhline(y=0, color='b', linestyle='--')
    plt.title('Violin Plots per Class')
    plt.xlabel('Feature (Node)')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.ylim([-11, 11])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    
def corr_mtrx(corr, save_path=None):
    """
    Plot the correlation matrix.
    Args:
        corr (pd.DataFrame): DataFrame containing the correlation matrix
        save_path (str): Path to save the image
    """
    # Plot the correlation between the features
    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, vmax=.3, center=0, cmap='mako',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()