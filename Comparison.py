from random import uniform
import sumolib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

net = sumolib.net.readNet('csr.net.xml')
G = nx.DiGraph()
node_positions = {}

for node in net.getNodes():
    node_id = node.getID()
    G.add_node(node_id)
    node_positions[node_id] = node.getCoord()

edge_map = {}
for edge in net.getEdges():
    from_node = edge.getFromNode().getID()
    to_node = edge.getToNode().getID()
    edge_id = edge.getID()
    weight = edge.getLength()  

    G.add_edge(from_node, to_node, weight=weight)
    edge_map[(from_node, to_node)] = edge_id

def heuristic(n):
    x1, y1 = node_positions[n]
    x2, y2 = node_positions[stop_node]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def get_neighbors(v):
    if v in G:
        return [(neighbor, G[v][neighbor]['weight']) for neighbor in G[v]]
    return []

from random import uniform, shuffle

def heuristic_randomised(n):
    """Base heuristic + small random noise (simulates traffic unpredictability)."""
    x1, y1 = node_positions[n]
    x2, y2 = node_positions[stop_node]
    base_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    noise = uniform(-0.1, 0.1) * base_distance  # -10% to +10% noise
    return base_distance + noise

def get_neighbors_randomized(v):
    """Randomize neighbor processing order."""
    neighbors = get_neighbors(v)
    shuffle(neighbors)
    return neighbors

def astaralgo_randomised(start_node, stop_node):
    open_set = set([start_node])
    closed_set = set()
    cost = {start_node: 0}
    parents = {start_node: None}

    while open_set:
        n = min(open_set, key=lambda node: cost[node] + heuristic_randomised(node) + uniform(0, 1e-3))

        if n == stop_node:
            path_edges = []
            while parents[n] is not None:
                prev = parents[n]
                path_edges.append(edge_map[(prev, n)])
                n = prev
            path_edges.reverse()
            return path_edges

        open_set.remove(n)
        closed_set.add(n)

        for m, weight in get_neighbors_randomized(n):
            if m in closed_set:
                continue

            tentative_cost = cost[n] + weight
            if m not in open_set:
                open_set.add(m)
                parents[m] = n
                cost[m] = tentative_cost
            elif tentative_cost < cost[m]:
                cost[m] = tentative_cost
                parents[m] = n

    print("Path does not exist!")
    return None


def astaralgo(start_node, stop_node):
    open_set = set([start_node])
    closed_set = set()
    cost = {start_node: 0}
    parents = {start_node: None}

    while open_set:
        n = min(open_set, key=lambda node: cost[node] + heuristic(node))

        if n == stop_node:
            path_edges = []
            while parents[n] is not None:
                path_edges.append(edge_map[(parents[n], n)])
                n = parents[n]
            path_edges.reverse()
            print("Path found: {}".format(path_edges))
            return path_edges

        open_set.remove(n)
        closed_set.add(n)

        for m, weight in get_neighbors(n):
            if m in closed_set:
                continue

            tentative_cost = cost[n] + weight
            if m not in open_set:
                open_set.add(m)
                parents[m] = n
                cost[m] = tentative_cost
            elif tentative_cost < cost[m]:
                cost[m] = tentative_cost
                parents[m] = n

    print("Path does not exist!")
    return None

start_node = '1779066710'
stop_node = '3779336540'
path_edges_rand = astaralgo_randomised(start_node, stop_node)
path_edges_det = astaralgo(start_node, start_node)

import matplotlib.pyplot as plt

# Run A* comparisons
num_trials = 100
random_costs = []
deterministic_costs = []

def calculate_total_cost(path_edges):
    """Calculate the total cost (sum of weights) for a path."""
    if not path_edges:
        return float('inf')
    total = 0
    for edge_id in path_edges:
        edge = net.getEdge(edge_id)
        total += edge.getLength()
    return total

def apply_random_traffic():
    for edge in net.getEdges():
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        base_length = edge.getLength()
        # Apply traffic: vary between 90% to 150% of normal
        traffic_factor = uniform(0.9, 1.5)
        weight = base_length * traffic_factor
        if G.has_edge(from_node, to_node):
            G[from_node][to_node]['weight'] = weight

for i in range(num_trials):
    # apply_random_traffic()
    # Randomized A*
    rand_path = astaralgo_randomised(start_node, stop_node)
    rand_cost = calculate_total_cost(rand_path)
    random_costs.append(rand_cost)

    # Deterministic A*
    det_path = astaralgo(start_node, stop_node)
    det_cost = calculate_total_cost(det_path)
    deterministic_costs.append(det_cost)

# Plotting the comparison
plt.figure(figsize=(10, 6))
plt.plot(deterministic_costs, label='Deterministic A*', marker='o', color='blue')
plt.plot(random_costs, label='Randomised A*', marker='x', color='orange')
plt.title("Cost Comparison: Deterministic vs Randomised A*")
plt.xlabel("Trial Number")
plt.ylabel("Total Path Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert lists to numpy arrays
random_costs_np = np.array(random_costs)
deterministic_costs_np = np.array(deterministic_costs)

# Mean
mean_rand = np.mean(random_costs_np)
mean_det = np.mean(deterministic_costs_np)

# Median
median_rand = np.median(random_costs_np)
median_det = np.median(deterministic_costs_np)

# Standard Deviation
std_rand = np.std(random_costs_np)
std_det = np.std(deterministic_costs_np)

# Print metrics
print("\n--- A* Cost Statistics ---")
print(f"Deterministic A*: Mean={mean_det:.2f}, Median={median_det:.2f}, Std Dev={std_det:.2f}")
print(f"Randomised   A*: Mean={mean_rand:.2f}, Median={median_rand:.2f}, Std Dev={std_rand:.2f}")

plt.figure(figsize=(12, 6))

plt.hist(deterministic_costs_np, bins=10, alpha=0.6, label='Deterministic A*', color='blue')
plt.hist(random_costs_np, bins=10, alpha=0.6, label='Randomised A*', color='orange')

plt.title("Histogram of Total Path Costs")
plt.xlabel("Total Path Cost")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([deterministic_costs_np, random_costs_np], labels=["Deterministic A*", "Randomised A*"])
plt.title("Boxplot Comparison of Path Costs")
plt.ylabel("Total Path Cost")
plt.grid(True)
plt.tight_layout()
plt.show()

