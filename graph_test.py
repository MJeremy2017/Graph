import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hclust
import pandas as pd
from scipy import stats
from collections import Counter
from operator import itemgetter

STEPS = 5000
NUM_NODES = 100

ba = nx.barabasi_albert_graph(NUM_NODES, 5)
nx.draw_networkx(ba, node_size=10)
# degree distribution
plt.figure()
density = stats.gaussian_kde(nx.degree(ba).values())

# ba_pg = nx.pagerank(ba)
# print sorted(ba_pg.items(), key=itemgetter(1), reverse=True)


def graph_2_dict(gh):
    gh_dict = {}
    for v, edg in gh.adjacency_iter():
        gh_dict[v] = edg.keys()
    return gh_dict


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


ba_dict = graph_2_dict(ba)
path, end_node = random_walk(ba_dict)

nodes_count = Counter(path)
print sorted(nodes_count.items(), key=itemgetter(1), reverse=True)
ba_pg = nx.pagerank(ba)
print sorted(ba_pg.items(), key=itemgetter(1), reverse=True)

# plot the density
nodes_count_x = nodes_count.keys()
nodes_count_y = [float(v)/STEPS for v in nodes_count.values()]

plt.plot(nodes_count_x, nodes_count_y, label='Random Walk')
plt.plot(ba_pg.keys(), ba_pg.values(), label='PageRank', color='orange', linestyle='dashed')
plt.legend(loc='best')
plt.title('Random Walk vs PageRank (steps=5000)')

# --------------------- community detection ---------------------- #

# transform the graph into a link matrix
# transform into another matrix using Jaccard Similarity
# clustering using hierarchical clustering


def jaccard_distance(l1, l2):
    return len(np.intersect1d(l1, l2))/float(len(np.union1d(l1, l2)))


J_mat = np.zeros((NUM_NODES, NUM_NODES))

for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        J_mat[i, j] = jaccard_distance(ba_dict[i], ba_dict[j])


Z = hclust.linkage(J_mat, 'ward')

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
hclust.dendrogram(Z)

# show last p merged clusters
plt.title('Hierarchical Clustering Dendrogram(last 8 merged clusters)')
plt.xlabel('sample index')
plt.ylabel('distance')
hclust.dendrogram(Z, truncate_mode='lastp', p=8)  # leaf is the size of each cluster

# get number of clusters
cutree = hclust.cut_tree(Z, 5)
# assign membership to a dict
membership = {}
for i in range(8):
    membership[i] = np.where(cutree == i)[0]

print membership

# ba_dict to csv
ba_arr = []
for source, targets in ba_dict.items():
    for target in targets:
        ba_arr.append([source, target])

ba_arr = np.array(ba_arr)

pd.DataFrame(ba_arr, columns=['source', 'target']).to_csv('ba_test.csv', index=False)
pd.DataFrame({'id': range(100), 'class': cutree.tolist()}).to_csv('ba_clustering.csv', index=False)

