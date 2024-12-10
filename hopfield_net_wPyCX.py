import random as rd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.filters import threshold_otsu

import networkx as nx
import numpy as np

import pycxsimulator
import matplotlib
matplotlib.use('TkAgg')

# imprints patterns


def imprint_patterns(G, pattern_list, synapse_wf):
    num_patterns = len(pattern_list)
    num_nodes = G.number_of_nodes()

    # Flatten all patterns
    flattened_patterns = [pat.flatten() for pat in pattern_list]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Compute weight using flattened patterns
            weight = synapse_wf * sum(flattened_patterns[k][i] * flattened_patterns[k][j]
                                      for k in range(num_patterns))
            G[i][j]['weight'] = weight
            G[j][i]['weight'] = weight


# generate random pattern
def generate_random_pattern(n):
    dimension = int(n**0.5)
    pat = np.random.choice([-1, 1], size=n)
    return np.reshape(pat, (dimension, dimension))


# compute for the max similarity percentage in a patterns group
def compute_error_percentage(states, initial_patterns):
    percetages = []

    for pattern in initial_patterns:
        percetages.append(
            100 - (sum(states.astype(int) == pattern.flatten())/states.size)*100)
        pattern = pattern*(-1)
        percetages.append(
            100 - (sum(states.astype(int) == pattern.flatten())/states.size)*100)

    return min(percetages)


# add noise to matrix
def add_noise(matrix, noise_level):
    noisy_matrix = matrix.copy()

    num_elements = matrix.size
    num_flips = int(noise_level * num_elements)

    flip_indices = np.random.choice(num_elements, num_flips, replace=False)
    row_indices, col_indices = np.unravel_index(flip_indices, matrix.shape)

    noisy_matrix[row_indices, col_indices] = np.random.choice(
        [1, -1], size=num_flips, replace=True)

    return noisy_matrix


# computes for hopfield network
def compute_energy(states):
    global hopfield_net
    energy = 0
    for i in hopfield_net.nodes():
        for j in hopfield_net.neighbors(i):
            energy -= 0.5 * \
                hopfield_net[i][j]['weight'] * states[i] * states[j]
    return energy


# prints image
def show_image(states):
    dimension = int(len(states)**0.5)
    img = np.reshape(states, (dimension, dimension))
    plt.imshow(img, cmap='binary', interpolation='nearest')
    plt.show()


def set_params(num_nodes_, num_patterns_, synapse_wf_=1, noise_level_=0, density_=1, will_comp_energy_=False):
    """

    """
    global num_nodes, patterns, initial_pattern, synapse_wf, noise_level, density, will_comp_energy
    patterns = []
    num_nodes = num_nodes_
    for i in range(num_patterns_):
        if i == 0:
            initial_pattern = generate_random_pattern(num_nodes)
            patterns.append(initial_pattern)
        else:
            patterns.append(generate_random_pattern(num_nodes))

    synapse_wf = synapse_wf_
    noise_level = noise_level_
    density = density_
    will_comp_energy = will_comp_energy_


def initialize():
    global hopfield_net, patterns, states, density, noise_level, synapse_wf

    # initialize graph edge connection
    max_edges = (num_nodes * (num_nodes - 1)) // 2
    num_edges = int(density * max_edges)

    possible_edges = [(i, j) for i in range(num_nodes)
                      for j in range(i + 1, num_nodes)]
    selected_edges = rd.sample(possible_edges, num_edges)

    hopfield_net = nx.Graph()
    hopfield_net.add_nodes_from(range(num_nodes))
    hopfield_net.add_edges_from(selected_edges)

    # initialize edge weights
    for (i, j) in hopfield_net.edges():
        hopfield_net[i][j]['weight'] = 0

    imprint_patterns(hopfield_net, patterns, synapse_wf)

    # initialize node states
    states = add_noise(initial_pattern, noise_level).flatten()


def observe():
    global states, num_nodes

    dimension = int(num_nodes**0.5)
    img = np.reshape(states, (dimension, dimension))

    plt.clf()
    plt.imshow(img, cmap='binary', interpolation='nearest')
    plt.show()


def update():
    global hopfield_net, states, will_comp_energy, energy_list

    # SYNCHRONOUS UPDATE
    new_states = states.copy()
    for i in hopfield_net.nodes():
        net_input = sum(hopfield_net[i][j]['weight'] * states[j]
                        for j in hopfield_net.neighbors(i))
        new_states[i] = int(np.sign(net_input))
        if new_states[i] == 0:
            new_states[i] = 1

        if will_comp_energy:
            energy_list.append(compute_energy(new_states))

    states = new_states.copy()

    # # ASYNCHRONOUS RANDOM UPDATE
    # i = np.random.choice(hop_network.nodes())
    # net_input = sum(hop_network[i][j]['weight'] * states[j]
    #                 for j in hop_network.neighbors(i))
    # states[i] = int(np.sign(net_input))
    # if states[i] == 0:
    #     states[i] = 1

"""
Initialize parameters here:
- to access the list of energy values per iteration of the node, use the variable energy_list
"""
# set_params(num_nodes_, num_patterns_, synapse_wf_=1, noise_level_=0, density_=1, will_comp_energy_=False)
set_params(10*10, 2, noise_level_=0.5)

pycxsimulator.GUI().start(func=[initialize, observe, update])
