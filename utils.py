#  IMPORTS
import tensorflow as tf
import pickle
import numpy as np
import random
import os

import tensorflow as tf
from graph_nets import graphs, utils_np, modules, utils_tf, blocks
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import sonnet as snt

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# encode action as one-hot encoded action
def one_hot_encoded_action(action, n_actions):
    return np.eye(n_actions, dtype=int)[action]

# load pickle file from file
def load_pickle(file):
    with open((file), 'rb') as f:
        data = pickle.load(f)
    return data

# chose a random start and target position in the given maze
def choose_random_start_and_target(maze):
    empty_cells = [(col, row) for col in range(maze.shape[1]) for row in range(maze.shape[0]) if maze[row, col] == 0]
    rnd_start_target = random.sample(empty_cells, 2)  # return random starting position in maze
    rnd_start = rnd_start_target[0]
    rnd_target = rnd_start_target[1]
    return rnd_start, rnd_target





def make_mlp_model(hidden_dim=10, out_dim=10):
    """Instantiates a new MLP with Layer Norm.
    The parameters of each new MLP are not shared with others generated by
    this function.
    Returns:
      A Sonnet module which contains the MLP and LayerNorm.
    """
    return snt.Sequential([
        snt.Linear(hidden_dim),
        tf.nn.relu,
        snt.Linear(hidden_dim),
        snt.LayerNorm(),
        tf.nn.relu,
        snt.Linear(out_dim)
    ])

# plot networkx graph
def plot_graph_networkx(graph, ax, colors=None, use_unique_colors=True, use_pos_node=False, pos=None, title=None):
    node_labels = {node: "{}".format(np.round(data["features"][:], 2))
                   for node, data in graph.nodes(data=True)
                   if data["features"] is not None}
    edge_labels = {(sender, receiver): "{}".format(data["features"][:])
                   for sender, receiver, data in graph.edges(data=True)
                   if data["features"] is not None}
    global_label = ("{}".format(graph.graph["features"][:])
                    if graph.graph["features"] is not None else None)

    if use_pos_node == False and pos == None:
        pos = nx.spring_layout(graph)
    elif use_pos_node == False:
        pos = pos
    elif use_pos_node == True:
        pos = nx.get_node_attributes(graph, name='features')

    # make the label such that it is not printend right onto the node but a slight offset in the y dimenion.
    pos_higher = {}
    offset = 0.1
    for i,j in pos.items():
        pos_higher[i] = (float(j[0]), float(j[1])+offset)

    print('node labels: ', node_labels)
    print('edge labels: ', edge_labels)

    if use_unique_colors:
        colors = cm.rainbow(np.linspace(0, 1, len(graph.nodes)))

    nx.draw_networkx(graph, pos, ax=ax, node_color=colors, with_labels=False, labels=None)
    # nx.draw_networkx(graph, pos, ax=ax, node_color=colors, with_labels=False, labels=None, connectionstyle='arc3, rad = 0.4')
    nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=node_labels)

    # edit pos to print all edge labels
    pos2 = pos.copy()
    cnt = 0
    for i in range(len(pos2), 2 * (len(pos2)) - 1):
        pos2[i] = pos2[cnt]
        cnt += 1

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos2, edge_labels, ax=ax, label_pos=0.2)

    if global_label:
        plt.text(0.05, 0.95, global_label, transform=ax.transAxes)

    if title:
        ax.set_title(title)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    return pos

# plot graphs tuple
def plot_graphs_tuple_np(graphs_tuple, pos=None):
    networkx_graphs = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
    num_graphs = len(networkx_graphs)
    _, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))
    if num_graphs == 1:
        axes = axes,
    for graph, ax in zip(networkx_graphs, axes):
        plot_graph_networkx(graph, ax, pos)

# plot multiple graph tuples
def plot_compare_graphs(graphs_tuples, labels, color, use_pos_node=False, pos=None):
    num_graphs = len(graphs_tuples)
    _, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))
    if num_graphs == 1:
        axes = axes,

    for name, graphs_tuple, ax in zip(labels, graphs_tuples, axes):
        graph = utils_np.graphs_tuple_to_networkxs(graphs_tuple)[0]
        plot_graph_networkx(graph, ax, colors=color, use_pos_node=use_pos_node, pos=pos)
        ax.set_title(name)