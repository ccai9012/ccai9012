"""
Visualization Utilities Module
=============================

This module provides a comprehensive set of visualization tools for data science and machine learning applications.
It includes functions for visualizing neural network architectures, training processes, word embeddings,
geospatial data, and various other types of data visualizations.

The module is organized into several categories:
- Neural network visualization: Functions for visualizing MLP architectures and training metrics
- Word embedding visualization: Tools for visualizing word embeddings and bias in language models
- Geospatial visualization: Functions for creating maps, heatmaps, and plotting geo-located data points
- Word cloud visualization: Utilities for generating word clouds from text data
- General data visualization: Additional visualization tools for various data types

"""
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

import folium
from folium.plugins import HeatMap
import random

from collections import Counter, defaultdict
import itertools

# ==========================
# Neural network visualization
# ==========================

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

# ==========================
# Word embedding visualization
# ==========================

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


def plot_bar_bias(df, word_col='word', bias_col='gender_direction', top_n=10,
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

def occupation_comparison(df, occ_fel_percent):
    """
    Plot a scatter plot comparing gender bias scores with real-world female percentages in different occupations.

    This function creates an interactive scatter plot using Plotly that shows the relationship
    between gender bias in word embeddings and the actual percentage of females in various occupations.

    Args:
        df (pd.DataFrame): DataFrame containing occupation words and their gender direction scores.
            Must include columns: 'word' and 'gender_direction'.
        occ_fel_percent (dict): Dictionary mapping occupation words to their real-world female percentage
            (as decimal values between 0 and 1).

    Returns:
        None: Displays the interactive Plotly scatter plot.
    """

    df['female_percent'] = df['word'].map(occ_fel_percent)

    # Drop rows where female percentage is not available (NaN)
    df_f = df.dropna(subset=['female_percent'])

    # Normalize gender_direction for color mapping
    color_scale = df_f['gender_direction']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_f['gender_direction'],
        y=df_f['female_percent'],
        mode='markers+text',
        text=df_f['word'],
        textposition='top center',
        marker=dict(
            size=12,
            color=color_scale,  # Color by bias score
            colorscale='RdBu',  # Red to Blue diverging scale
            cmin=-max(abs(color_scale)),
            cmax=max(abs(color_scale)),
            colorbar=dict(
                title="Gender direction",
                tickformat=".2f"
            ),
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertemplate='<b>%{text}</b><br>Bias Score: %{x:.3f}<br>Female %: %{y:.1%}<extra></extra>'
    ))

    # Add vertical line at x=0
    fig.add_shape(
        type='line',
        x0=0, x1=0,
        y0=min(df_f['female_percent']),
        y1=max(df_f['female_percent']),
        line=dict(color='gray', dash='dash')
    )

    fig.update_layout(
        title='Gender Bias Score vs Female Percentage in Occupations',
        xaxis_title='Gender Direction (sim("he") - sim("she"))',
        yaxis_title='Percentage of Females in Occupation',
        template='plotly_white',
        width=850,
        height=500
    )

    fig.show()

# ==========================
# Geospatial visualization
# ==========================

def plot_heatmap(df, target_col='price'):
    """
    Create a folium heatmap visualization from geographical data with values as weights.

    This function takes a DataFrame with latitude and longitude coordinates and creates
    a heatmap where the intensity is determined by the values in the target column.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'latitude', 'longitude', and target_col.
        target_col (str, optional): Column name to use for heatmap intensity values. Defaults to 'price'.

    Returns:
        folium.Map: Interactive folium map with heatmap layer.
    """
    # Calculate map center coordinates
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    # Initialize Folium map centered at mean latitude and longitude
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    # Prepare heatmap data as list of [lat, lon, weight]
    heat_data = [[row['latitude'], row['longitude'], row[target_col]] for idx, row in df.iterrows()]

    # Add heatmap layer with price as weight
    HeatMap(
        heat_data,
        radius=15,  # radius of each heat point
        blur=10,  # blur intensity
        max_zoom=13,
        max_val=df[target_col].max(),
        min_opacity=0.3
    ).add_to(m)

    return m


import folium
import branca.colormap as cm


def plot_points(
        df,
        lat_col='latitude',
        lon_col='longitude',
        value_col='weighted_score',
        map_center=None,
        zoom=15,
        colormap_colors=['blue', 'green', 'yellow', 'red'],
        colormap_caption=None,
        radius=7,
        fill_opacity=0.7,
):
    """
    Plot points on a folium map, color-coded by a target column.

    Args:
        df (pd.DataFrame): DataFrame containing coordinates and value_col.
        lat_col (str): Name of latitude column.
        lon_col (str): Name of longitude column.
        value_col (str): Column used to determine color.
        map_center (list/tuple): [lat, lon] for map center. If None, use mean coordinates.
        zoom (int): Initial zoom level.
        colormap_colors (list): List of colors for colormap.
        colormap_caption (str): Caption for color legend.
        radius (int): Radius of CircleMarkers.
        fill_opacity (float): Fill opacity of markers.

    Returns:
        folium.Map: Folium map object.
    """
    if map_center is None:
        map_center = [df[lat_col].mean(), df[lon_col].mean()]

    m = folium.Map(location=map_center, zoom_start=zoom, tiles='cartodbpositron')

    # Set min/max for colormap based on data
    vmin, vmax = df[value_col].min(), df[value_col].max()

    colormap = cm.LinearColormap(
        colors=colormap_colors,
        vmin=vmin,
        vmax=vmax,
        caption=colormap_caption or value_col
    )

    # Add points
    for _, row in df.iterrows():
        lat, lon, val = row[lat_col], row[lon_col], row[value_col]
        color = colormap(val)
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=f"{value_col}: {val:.2f}"
        ).add_to(m)

    colormap.add_to(m)

    return m

def plot_review_map(df, name_field = 'business_name', sentiment_field='overall_impression', map_center=[22.3193, 114.1694], zoom=12):
    """
    Plot Airbnb review locations on a map, color-coded by a given sentiment field.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'latitude', 'longitude', and the sentiment_field.
        name_field (str): Name of column in the DataFrame containing review locations.
        sentiment_field (str): The field to determine color (e.g., 'overall_impression', 'location_opinion').
        map_center (list): Map center as [lat, lon].
        zoom (int): Initial zoom level.

    Returns:
        folium.Map: Folium map object with plotted points.
    """
    # Define color mapping
    impression_color = {
        'positive': 'green',
        'neutral': 'gray',
        'negative': 'red'
    }

    # Create map
    m = folium.Map(location=map_center, zoom_start=zoom, tiles="CartoDB positron")

    # Plot each review
    for _, row in df.iterrows():
        lat, lon = row.get('latitude'), row.get('longitude')
        name = row.get(name_field, 'Unknown')
        sentiment = row.get(sentiment_field, 'neutral')
        color = impression_color.get(sentiment, 'gray')

        if pd.notna(lat) and pd.notna(lon):
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{name}, {sentiment_field}: {sentiment}"
            ).add_to(m)

    return m


def plot_review_heatmap(reviews, map_center=None, zoom_start=11):
    """
    Create a heatmap visualization of review locations using folium.

    This function takes a list of review dictionaries containing geographic coordinates
    and plots them as a heatmap on an interactive map.

    Args:
        reviews (list): List of review dictionaries, each containing at least 'latitude'
            and 'longitude' keys.
        map_center (tuple, optional): (latitude, longitude) tuple to center the map. If None,
            the center is calculated as the average of all review locations.
        zoom_start (int, optional): Initial zoom level for the map. Defaults to 11.

    Returns:
        folium.Map: Interactive folium map with heatmap layer, or None if no valid data is provided.
    """
    # Prepare location points
    heat_data = [
        [review['latitude'], review['longitude']]
        for review in reviews
        if review['latitude'] is not None and review['longitude'] is not None
    ]

    if not heat_data:
        print("No valid location data to plot.")
        return

    # If no center provided, use mean location
    if map_center is None:
        avg_lat = sum(p[0] for p in heat_data) / len(heat_data)
        avg_lon = sum(p[1] for p in heat_data) / len(heat_data)
        map_center = (avg_lat, avg_lon)

    # Create folium map
    m = folium.Map(location=map_center, zoom_start=zoom_start)
    HeatMap(heat_data, radius=10, blur=15).add_to(m)

    return m

def plot_poi_sampled(pois, center=(36.1699, -115.1398), zoom=4, sample_size=1000):
    """
    Plot a random sample of business coordinates on a folium map.

    Args:
        pois (list): List of business dicts with lat/lon
        center (tuple): Initial center of the map
        zoom (int): Initial zoom level
        sample_size (int): Number of businesses to sample for plotting

    Returns:
        folium.Map
    """
    sampled = random.sample(pois, min(sample_size, len(pois)))

    m = folium.Map(location=center, zoom_start=zoom)

    for b in sampled:
        folium.CircleMarker(
            location=(b['latitude'], b['longitude']),
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"{b['name']} ({b['city']})"
        ).add_to(m)

    return m

import osmnx as ox
import geopandas as gpd
import numpy as np
import folium
from shapely.geometry import Point
from pyproj import Transformer
from scipy.spatial import KDTree


def sample_street_points_map(bbox, dist=100, min_distance_m=50, zoom=15):
    """
    Sample points along streets within a bounding box and visualize them on a folium map.

    Args:
        bbox (tuple): (lat_min, lon_min, lat_max, lon_max)
        dist (float): approximate distance between sampled points along lines (in meters)
        min_distance_m (float): minimum distance between points after KDTree filtering
        zoom (int): initial zoom level for the folium map

    Returns:
        folium.Map: folium map with sampled points
        list of tuple: sampled points as (lat, lon)
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    map_center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    # --------------------------
    # Function to sample points along a linestring at approx. regular intervals
    def sample_points_along_line(line, dist):
        length = line.length
        points = [line.interpolate(d) for d in np.arange(0, length, dist)]
        return points

    # KDTree filtering function
    def filter_points_kdtree(points_proj, min_distance_m):
        if not points_proj:
            return []
        xy = np.array([[pt.x, pt.y] for pt in points_proj])
        tree = KDTree(xy)
        kept_idx = []
        visited = np.zeros(len(xy), dtype=bool)
        for i in range(len(xy)):
            if not visited[i]:
                kept_idx.append(i)
                neighbors = tree.query_ball_point(xy[i], r=min_distance_m)
                visited[neighbors] = True
        return [points_proj[i] for i in kept_idx]

    # --------------------------
    # Download street network graph within bounding box
    G = ox.graph_from_bbox(
        bbox,
        network_type='all'
    )

    # Project graph to a metric CRS (meters)
    G_proj = ox.project_graph(G)

    # Extract edges as GeoDataFrame
    edges = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)

    # Sample points from all edges
    sampled_points_proj = []
    for line in edges.geometry:
        sampled_points_proj.extend(sample_points_along_line(line, dist=dist))

    # Filter points to ensure minimum spacing
    filtered_points_proj = filter_points_kdtree(sampled_points_proj, min_distance_m=min_distance_m)

    # Convert filtered points from projected CRS back to latitude/longitude
    project_crs = edges.crs
    to_latlon = Transformer.from_crs(project_crs, "EPSG:4326", always_xy=True)
    sampled_points_latlon = [to_latlon.transform(pt.x, pt.y)[::-1] for pt in filtered_points_proj]  # (lat, lon)

    # Create folium map
    m = folium.Map(location=map_center, zoom_start=zoom, tiles="cartodbpositron")

    # Add bounding box rectangle
    bounding_box = [(lat_min, lon_min), (lat_min, lon_max),
                    (lat_max, lon_max), (lat_max, lon_min), (lat_min, lon_min)]
    folium.PolyLine(locations=bounding_box, color="red", weight=3, opacity=0.8).add_to(m)

    # Add CircleMarkers for sampled points
    for lat, lon in sampled_points_latlon:
        folium.CircleMarker(location=[lat, lon], radius=1, color='blue',
                            fill=True, fill_opacity=0.5).add_to(m)

    print(f"In total {len(sampled_points_latlon)} points sampled.")
    return m, sampled_points_latlon


# ==========================
# Word cloud visualization
# ==========================

from wordcloud import WordCloud


def plot_wordclouds(df, target_col='overall_impression', tag_col='decision_tags', delimiter=','):
    """
    Plot word clouds of decision_tags grouped by overall_impression categories.

    Args:
        df (pd.DataFrame): DataFrame containing overall impression and tags columns.
        target_col (str): Column name for overall impression (categorical).
        tag_col (str): Column name for decision tags.
        delimiter (str): Delimiter separating multiple tags in one entry.
    """
    overall_values = df[target_col].dropna().unique()
    num_categories = len(overall_values)

    plt.figure(figsize=(8, 4 * num_categories))

    for i, val in enumerate(sorted(overall_values)):
        # Filter rows for this overall impression
        subset = df[df[target_col] == val]
        # Combine all tags, split and clean
        all_tags = subset[tag_col].dropna().astype(str)
        tags_list = all_tags.str.split(delimiter).explode().str.strip()
        text = ' '.join(tags_list)

        wc = WordCloud(
            width=600, height=300,
            background_color='white',
            collocations=False
        ).generate(text)

        plt.subplot(num_categories, 1, i + 1)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Decision Tags Word Cloud\nOverall Impression: {val}")

    plt.tight_layout()
    plt.show()


def plot_wordclouds_by_aspect_opinion(df, aspect_cols, comment_col_prefix='{}_comment'):
    """
    Plot word clouds of comments grouped by opinions for each given aspect.

    Args:
        df (pd.DataFrame): DataFrame containing opinion columns and comment columns.
        aspect_cols (list of str): List of aspect column names, e.g. ['location_opinion', 'facility_opinion', 'host_opinion'].
        comment_col_prefix (str): Pattern to get comment column name from aspect name, default '{}_comment'.
                                  For example, if aspect is 'location_opinion', comment_col is 'location_comment'.
    """
    for aspect in aspect_cols:
        opinions = df[aspect].dropna().unique()
        num_opinions = len(opinions)

        plt.figure(figsize=(6, 3 * num_opinions))
        plt.suptitle(f"Word Clouds of {aspect.replace('_', ' ').capitalize()} Comments by Opinion", fontsize=16)

        comment_col = comment_col_prefix.format(aspect.split('_')[0])  # e.g. 'location_comment'

        for i, opinion in enumerate(sorted(opinions)):
            # Select rows matching the opinion and drop missing comments
            comments = df[df[aspect] == opinion][comment_col].dropna().astype(str)
            text = ' '.join(comments)

            wc = WordCloud(
                width=600,
                height=300,
                background_color='white',
                collocations=False
            ).generate(text)

            plt.subplot(num_opinions, 1, i + 1)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{opinion.capitalize()}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        plt.show()


# ==========================
# Other visualization utilities
# ==========================

def plot_star_distribution(reviews):
    stars = [review['stars'] for review in reviews]

    plt.figure(figsize=(8, 5))
    plt.hist(stars, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', rwidth=0.8)
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Reviews")
    plt.title("Distribution of Yelp Review Star Ratings")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def viz_keywords_freq(results_df, column="materials", top_k=None, save_path=None):
    """
    Visualize the frequency of keywords in a specified column of a DataFrame.

    This function extracts keywords from a comma-separated column in the DataFrame,
    counts their frequency, and visualizes the results as a bar chart.

    Args:
        results_df (pd.DataFrame): DataFrame containing the column with comma-separated keywords.
        column (str, optional): Name of column containing comma-separated keywords. Defaults to "materials".
        top_k (int, optional): Number of top keywords to display. If None, all keywords are shown.
        save_path (str, optional): If provided, the plot is saved to this file path.

    Returns:
        collections.Counter: Counter object with keyword frequencies.
    """
    all_keywords = []
    for entry in results_df[column].dropna():
        keywords = [kw.strip().lower() for kw in entry.split(",") if kw.strip()]
        all_keywords.extend(keywords)

    counter = Counter(all_keywords)
    most_common = counter.most_common(top_k)

    labels, counts = zip(*most_common) if most_common else ([], [])

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts)
    plt.xticks(rotation=34, ha="right")
    plt.title(f"Frequency of {column} keywords")
    plt.ylabel("Count")
    plt.tight_layout()

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(count),
                 ha='center', va='bottom', fontsize=10)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    return counter


def plot_cooccurrence_heatmap(results_df, column="materials", top_k=10, save_path=None):
    """
    Plot a co-occurrence heatmap of material keywords that frequently appear together.

    Args:
        results_df (pd.DataFrame): DataFrame containing a column with comma-separated material keywords.
        column (str): Name of the column containing keywords (default: "materials").
        top_k (int): Number of top frequent keywords to include in the heatmap.

    Returns:
        pd.DataFrame: Co-occurrence matrix of the top_k keywords.
    """
    # Step 1: Count frequency of all individual keywords
    keyword_counter = Counter()
    for entry in results_df[column].dropna():
        keywords = [kw.strip().lower() for kw in entry.split(",") if kw.strip()]
        keyword_counter.update(keywords)

    # Select top_k most frequent keywords
    top_keywords = [kw for kw, _ in keyword_counter.most_common(top_k)]

    # Step 2: Initialize a co-occurrence matrix as a nested dictionary
    co_matrix = defaultdict(lambda: defaultdict(int))

    # Step 3: Iterate through each row and count co-occurrences
    for entry in results_df[column].dropna():
        keywords = [kw.strip().lower() for kw in entry.split(",") if kw.strip()]
        # Keep only keywords that are in the top_k list
        keywords = [kw for kw in keywords if kw in top_keywords]
        # Count pairwise combinations
        for kw1, kw2 in itertools.combinations(set(keywords), 2):
            co_matrix[kw1][kw2] += 1
            co_matrix[kw2][kw1] += 1
        # Optionally count self-co-occurrence
        for kw in keywords:
            co_matrix[kw][kw] += 1

    # Step 4: Convert the nested dictionary to a DataFrame
    co_df = pd.DataFrame(co_matrix).fillna(0).astype(int)
    co_df = co_df.reindex(index=top_keywords, columns=top_keywords)

    # Step 5: Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(co_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Co-occurrence Heatmap of Top {top_k} Material Keywords")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

    return co_df