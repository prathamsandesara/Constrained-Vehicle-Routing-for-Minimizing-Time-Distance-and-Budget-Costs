from random import randint
import sumolib
import networkx as nx
import matplotlib.pyplot as plt

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
path_edges = astaralgo(start_node, stop_node)

def write_routes(path_edges):
    if path_edges is None:
        print("No valid path found. Cannot generate route file.")
        return

    with open("test.rou.xml", "w") as f:
        f.write("""<routes>
    <vType id="car" accel="1.0" decel="5.0" sigma="0.5" length="5" maxSpeed="50"/>
    <vehicle id="1000" type="car" depart="1" color="1,0,0">
    <route edges="{}"/></vehicle>
</routes>""".format(" ".join(path_edges)))

    print("Route file `test.rou.xml` generated.")

write_routes(path_edges)

plt.figure(figsize=(15, 25))
nx.draw(G, pos=node_positions, with_labels=True, node_color='lightblue', edge_color='gray',
        node_size=250, font_size=4, arrows=True, connectionstyle="arc3, rad=0.02")

if path_edges:
    path_nodes = []
    for (from_node, to_node), edge_id in edge_map.items():
        if edge_id in path_edges:
            path_nodes.append((from_node, to_node))

    nx.draw_networkx_edges(G, pos=node_positions, edgelist=path_nodes, edge_color='green', width=2)

plt.title("SUMO Network Visualization with Highlighted Path")
plt.show()
