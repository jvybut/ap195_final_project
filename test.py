import networkx as nx

# Example graph
G = nx.erdos_renyi_graph(10, 0.5)

# Calculate density
density = nx.density(G)
print(f"Density of the graph: {density:.2f}")
