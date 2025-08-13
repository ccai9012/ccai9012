import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

import torch
import pygraphviz as pgv
from IPython.display import Image


def draw_simple_mlp(
        input_size=8,
        hidden_size=64,
        output_size=1,
        hidden_display=5,
        # Number of hidden nodes to display (for clarity), actual hidden layer size remains hidden_size
        input_color='lightblue',
        hidden_color='lightgreen',
        output_color='lightcoral',
        layer_label_color='lightgrey',
        layout_prog='dot',
        filename='simple_mlp.png',
        figsize=(30, 10),
        dpi=100,
        nodesep=1.0,
        ranksep=4.0,
        fontsize=24
):
    """
    Draws a simple MLP (Multi-Layer Perceptron) architecture graph using pygraphviz.
    Parameters:
    - input_size: int, number of input layer nodes
    - hidden_size: int, total number of hidden layer nodes (used for labeling)
    - output_size: int, number of output layer nodes
    - hidden_display: int, number of hidden nodes to actually display (for visualization clarity)
    - input_color, hidden_color, output_color: colors for nodes of each layer
    - layer_label_color: color for the subgraph (layer) labels and borders
    - layout_prog: graphviz layout program to use (e.g., 'dot', 'neato')
    - filename: output image filename
    - figsize: tuple(width, height), size of the graph image
    - dpi: image resolution
    - nodesep: separation between nodes
    - ranksep: separation between layers (ranks)
    - fontsize: font size for labels

    Returns:
    - IPython.display.Image object of the generated graph image
    """

    G = pgv.AGraph(strict=False, directed=True)

    # Create input layer nodes
    input_nodes = [f'in{i + 1}' for i in range(input_size)]
    for node in input_nodes:
        G.add_node(node, shape='circle', style='filled', fillcolor=input_color)

    # Create hidden layer nodes for visualization (limited to hidden_display)
    hidden_display = min(hidden_display, hidden_size)
    hidden_nodes = [f'hid{i + 1}' for i in range(hidden_display)]
    for node in hidden_nodes:
        G.add_node(node, shape='circle', style='filled', fillcolor=hidden_color)

    # Create output layer nodes
    output_nodes = [f'out{i + 1}' for i in range(output_size)]
    for node in output_nodes:
        G.add_node(node, shape='circle', style='filled', fillcolor=output_color)

    # Add edges from input layer to hidden layer
    for i_node in input_nodes:
        for h_node in hidden_nodes:
            G.add_edge(i_node, h_node)

    # Add edges from hidden layer to output layer
    for h_node in hidden_nodes:
        for o_node in output_nodes:
            G.add_edge(h_node, o_node)

    # Define subgraph for input layer with label and styling
    with G.subgraph(name='cluster_input') as c:
        c.add_nodes_from(input_nodes)
        c.graph_attr.update(label=f'Input Layer ({input_size})', color=layer_label_color, style='dashed')
        c.graph_attr.update(rank='same')  # Keep nodes in the same rank (horizontal alignment)

    # Define subgraph for hidden layer with label and styling
    with G.subgraph(name='cluster_hidden') as c:
        c.add_nodes_from(hidden_nodes)
        c.graph_attr.update(label=f'Hidden Layer ({hidden_size})', color=layer_label_color, style='dashed')
        c.graph_attr.update(rank='same')

    # Define subgraph for output layer with label and styling
    with G.subgraph(name='cluster_output') as c:
        c.add_nodes_from(output_nodes)
        c.graph_attr.update(label=f'Output Layer ({output_size})', color=layer_label_color, style='dashed')
        c.graph_attr.update(rank='same')

    # Set global graph attributes for layout and style
    G.graph_attr.update(
        rankdir='LR',  # Layout direction Left to Right
        size=f'{figsize[0]},{figsize[1]}!',
        dpi=str(dpi),
        nodesep=str(nodesep),
        ranksep=str(ranksep),
        fontsize=str(fontsize),
        splines='line'  # Use straight lines for edges
    )

    # Render the graph to file
    G.draw(filename, prog=layout_prog)

    # Return the image for display in Jupyter Notebook
    return Image(filename)


def plot_loss_curve(
        train_losses,
        val_losses,
        num_epochs=None,
        title="Training and Validation Loss Curve",
        xlabel="Epoch",
        ylabel="MSE Loss",
        figsize=(8, 5)
):
    """
    Plot training and validation loss curves.

    Parameters:
    - train_losses: list or array of training loss values per epoch
    - val_losses: list or array of validation loss values per epoch
    - num_epochs: int, number of epochs (if None, inferred from length of train_losses)
    - title: str, plot title
    - xlabel: str, label for x-axis
    - ylabel: str, label for y-axis
    - figsize: tuple, figure size
    """
    if num_epochs is None:
        num_epochs = len(train_losses)

    plt.figure(figsize=figsize)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_tsne_words(tsne_results, valid_words, key_words, key_word_colors=None, key_word_labels=None):
    """
    Plot t-SNE results of word embeddings with key words highlighted by color.

    Args:
        tsne_results (np.ndarray): 2D array of shape (N, 2) with t-SNE coordinates.
        valid_words (list of str): List of words corresponding to embeddings.
        key_words (set or list): Words to highlight with special colors.
        key_word_colors (dict or None): Optional dict mapping keys to colors.
            Example: {'group1': 'red', 'group2': 'blue'}
        key_word_labels (dict or None): Optional dict mapping keys to labels for legend.
    """
    pio.renderers.default = 'notebook'  # or 'notebook_connected'
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 8))

    # Default color assignment
    colors = []
    for w in valid_words:
        if w in key_words:
            # If key_word_colors provided and word belongs to group, use color
            if key_word_colors:
                # find color by group, fallback to 'red'
                # Here assume key_words is a dict {group_name: set_of_words}
                found_color = 'red'
                for group, words_set in key_words.items():
                    if w in words_set:
                        found_color = key_word_colors.get(group, 'red')
                        break
                colors.append(found_color)
            else:
                colors.append('red')
        else:
            colors.append('gray')

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s=100, edgecolors='k', alpha=0.75)

    for i, word in enumerate(valid_words):
        fontweight = 'bold' if (word in key_words if isinstance(key_words, (set, list)) else any(word in s for s in key_words.values())) else 'normal'
        color = 'black'
        if key_word_colors and isinstance(key_words, dict):
            for group, words_set in key_words.items():
                if word in words_set:
                    color = key_word_colors.get(group, 'black')
                    break

        plt.text(tsne_results[i, 0]+0.5, tsne_results[i, 1]+0.5, word,
                 fontsize=11, fontweight=fontweight,
                 color=color)

    plt.title('t-SNE Visualization of Word Groups', fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)

    # Legend
    legend_elements = []
    if key_word_colors and key_word_labels:
        for group, color in key_word_colors.items():
            label = key_word_labels.get(group, group)
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                                          markerfacecolor=color, markersize=12, markeredgecolor='k'))
    else:
        # default legend entries
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Key words',
                   markerfacecolor='red', markersize=12, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Other words',
                   markerfacecolor='gray', markersize=12, markeredgecolor='k'),
        ]

    plt.legend(handles=legend_elements, loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_bar_bias(df, word_col='word', bias_col='bias_score', top_n=10,
                  neg_color='mediumvioletred', pos_color='royalblue',
                  title=None, xlabel=None):
    """
    Plot horizontal bar chart of top negative and positive bias scores.

    Args:
        df (pd.DataFrame): DataFrame containing words and bias scores.
        word_col (str): Column name for words.
        bias_col (str): Column name for bias scores.
        top_n (int): Number of top positive/negative bars to show.
        neg_color (str): Color for negative bars.
        pos_color (str): Color for positive bars.
        title (str or None): Plot title.
        xlabel (str or None): X-axis label.
    """

    top_neg = df[df[bias_col] < 0].nsmallest(top_n, bias_col)
    top_pos = df[df[bias_col] > 0].nlargest(top_n, bias_col)
    top_words = pd.concat([top_neg, top_pos]).sort_values(bias_col).reset_index(drop=True)

    colors = [neg_color if x < 0 else pos_color for x in top_words[bias_col]]

    fig = go.Figure(go.Bar(
        x=top_words[bias_col],
        y=top_words[word_col],
        orientation='h',
        marker_color=colors,
        hovertemplate=f'%{{y}}<br>Bias Score: %{{x:.3f}}<extra></extra>'
    ))

    fig.update_layout(
        title=title or f'Top {top_n} Negative and Positive Bias Scores',
        xaxis_title=xlabel or bias_col,
        yaxis=dict(autorange='reversed'),
        template='plotly_white',
        margin=dict(l=120, r=40, t=80, b=40),
        font=dict(size=14)
    )

    fig.show()
