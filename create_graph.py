import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

g = nx.Graph()

g.add_nodes_from(range(10))
nx.draw(g)

H = nx.path_graph(10)
nx.draw(H)

g.add_nodes_from(H)
nx.draw(g)

g.add_edge(1, 2)
nx.draw(g)

g.add_node('spam')
g.number_of_edges()
g.number_of_nodes()

print g.nodes()
print g.edges()

h = nx.DiGraph(g)
nx.draw(h)

print h.edges()

FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
nx.draw(FG)

# iterate graph
for n, b in FG.adjacency_iter():
    print n, b

for (u, v, d) in FG.edges(data='weight'):
    print (u, v, d)

g = nx.Graph()
g.add_edges_from([(1, 2), (3, 4)], color='yellow')
nx.draw(g)

DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
nx.draw(DG)
DG.degree(1, weight='weight')
DG.out_degree()
DG.in_degree()
DG.successors(2)  # next node
DG.neighbors(1)

# multi-edges graph
MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, .5), (1, 2, .75), (2, 3, .5)])
nx.draw(MG)
MG.edges()

er = nx.erdos_renyi_graph(100, 0.15)
nx.draw(er)

nx.degree(g)

nx.draw(g)
nx.draw_random(g)
nx.draw_circular(g)
nx.draw_spectral(g)

nx.draw_networkx(g, with_labels=True)

dg = nx.DiGraph()
dg.add_edges_from([(1, 2), [1, 3], [1, 4], [2, 4]])
nx.draw_networkx(dg, with_labels=True, arrows=.2)

pr = nx.pagerank(dg)
print pr
np.sum([v for k, v in pr.items()])

dg2 = nx.DiGraph()
dg2.add_weighted_edges_from([(1, 2, 100), (1, 3, 2.2), (1, 4, 2), (2, 4, 2.3)])
nx.draw_networkx(dg2, with_labels=True)
dg2.add_weighted_edges_from([(1, 5, 5)])


pr2 = nx.pagerank(dg2, max_iter=200)
print pr2

ba = nx.barabasi_albert_graph(100, 5)
nx.draw_networkx(ba)

ba_pg = nx.pagerank(ba)
print sorted(ba_pg.items(), key=itemgetter(1), reverse=True)
