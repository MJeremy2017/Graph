import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
import networkx as nx


facebook = np.loadtxt('facebook_combined.txt')
facebook[0, :]
facebook.shape

print 'number of start nodes', len(set(facebook[:, 0]))
print 'number of end nodes', len(set(facebook[:, 1]))

# transform the data set into a dict
nodes = set(facebook[:, 0])
nodes_2_dict = {}
for start_node in nodes:
    nodes_2_dict[start_node] = np.intersect1d(facebook[facebook[:, 0] == start_node][:, 1], list(nodes))

# remove null values
for k, v in nodes_2_dict.items():
    if len(v) == 0:
        del nodes_2_dict[k]


def random_walk(data, steps=5000):
    walk_path = np.zeros(steps)
    # randomly choose a starting node
    start_node = np.random.choice(data.keys())
    walk_path[0] = start_node
    for i in range(steps-1):
        if start_node not in data.keys():
            start_node = np.random.choice(data.keys())
        # print i, start_node
        end_node = np.random.choice(data[start_node])
        start_node = end_node
        walk_path[i+1] = start_node

    return walk_path, end_node


# start random walk and record the path matrix
REPS = 50000
STEPS = 100

# path_mat = np.zeros((REPS, STEPS))
end_nodes = []

for i in range(REPS):
    path, end_node = random_walk(nodes_2_dict, STEPS)
    # path_mat[i, :] = path
    end_nodes.append(end_node)

plt.figure()
plt.hist(facebook[:, 0], bins=200)
plt.figure()
plt.hist(end_nodes, bins=200)

# count nodes
nodes_count = Counter(end_nodes)
print sorted(nodes_count.items(), key=itemgetter(1), reverse=True)

nodes_count2 = [(k, len(v)) for k, v in nodes_2_dict.items()]
print sorted(nodes_count2, key=itemgetter(1), reverse=True)[:15]


# just walk once
path, end_node = random_walk(nodes_2_dict, steps=1000000)

# should look at the pagerank
DG = nx.DiGraph()
DG.add_edges_from(facebook)

# nx.draw_networkx(DG)
pg = nx.pagerank(DG)
print sorted(pg.items(), key=itemgetter(1), reverse=True)[:20]
nodes_count = Counter(path)
print sorted(nodes_count.items(), key=itemgetter(1), reverse=True)

top_pg = sorted(pg.items(), key=itemgetter(1), reverse=True)[:50]
top_rw = sorted(nodes_count.items(), key=itemgetter(1), reverse=True)[:50]

top_int = np.intersect1d([item[0] for item in top_pg], [item[0] for item in top_rw])
len(top_int)  # 28/50
