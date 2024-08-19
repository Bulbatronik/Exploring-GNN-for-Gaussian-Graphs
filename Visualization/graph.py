from pathlib import Path

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

def vis_graph(A: np.array, 
              node_weight: np.array, 
              label: str, 
              save_path: Path=None):
    """
    Visualize a graph with colored nodes and edges
    Args:
        A (numpy.ndarray): Adjacency matrix of the graph
        node_weight (numpy.ndarray): Weight of each node (initial observations)
        label (str): Label of the class
        save_path (str, optional): Path to save the visualization image. Defaults to None.
    """
    # Visualize one graph and color the nodes according to the feature values
    G=nx.from_numpy_array(A)

    plt.figure()
    plt.axis('off')
    nx.draw_networkx(G,
                    #pos=nx.circular_layout(G), 
                    pos=nx.spring_layout(G, seed=0),
                    node_size=300,
                    cmap='coolwarm',
                    font_size=10,
                    font_color='white',
                    node_color=node_weight,
                    width=1,
                    edge_cmap=plt.cm.Blues,
                    edge_color=[G[u][v]['weight'] for u, v in G.edges]                       
                    )
    plt.title(f'{label} class')
    if save_path:
        plt.savefig(save_path, transparent=True)
    plt.show()
    
def plot_graphs(A: np.array,
                x: np.array, 
                y: np.array, 
                num_graphs: int, 
                class_name: str,
                save_path: Path=None) -> None:
    """
    Plots the first `num_graphs` graphs from the dataset, with each graph's nodes
    colored according to the feature values.
    Args:
    A (numpy.ndarray): Adjacency matrices of the graphs
    x (numpy.ndarray): Feature values of the nodes
    y (numpy.ndarray): Labels of the graphs
    """
    class_id = 1 if class_name == 'Positive' else 0 # Use value to get indexes of the respective class
    ids = np.where(y==class_id)[0]
    
    num_rows = num_graphs**0.5
    assert num_rows.is_integer(), 'num_graphs should be a square' # for convenience 
    
    fig, axs = plt.subplots(int(num_rows), int(num_rows))
    fig.suptitle(f'{class_name} class')

    # Plot graphs
    for i, idx in enumerate(ids[-num_graphs:]):
        # If we plot one graph only, we have one object
        if num_graphs == 1: 
            ix = i
            ax = axs
        else: # else many objects
            ix = np.unravel_index(i, axs.shape)
            ax = axs[ix]
            
        ax.axis('off')
        G = nx.from_numpy_array(A[idx])
        nx.draw_networkx(G,
                        #pos=nx.circular_layout(G), 
                        pos=nx.spring_layout(G, seed=0),
                        with_labels=True,
                        cmap='coolwarm',
                        node_size=15,
                        font_size=4,
                        node_color=x[idx, :, 0],
                        ax=ax,
                        width=0.8,
                        edge_cmap=plt.cm.Blues,
                        edge_color=[G[u][v]['weight'] for u, v in G.edges]                       
                        )
    if save_path:
        plt.savefig(save_path, dpi=1000, transparent=True)
    plt.show()