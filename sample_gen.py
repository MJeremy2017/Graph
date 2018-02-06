import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# generate scores subject to normal distribution
# randomly assign links to scores
# according to the links, compute the pagerank probabilities
# according to probabilities, reassign links

NUM_MEN = 100
NUM_WOMEN = 100


def score_generate(size=None, mean=None, std=None):
    return np.random.normal(loc=mean, scale=std, size=size)


# make link assignment according to probabilities
# return a directed graph
def link_assign(data1, data2, probs=None, first=True):
    dg = nx.DiGraph()
    len1 = len(data1)
    len2 = len(data2)

    def softmax(d1, d2):
        prob1 = np.exp(d1)/np.sum(np.exp(d1))
        prob2 = np.exp(d2)/np.sum(np.exp(d2))
        return prob1, prob2

    if first:
        prob1, prob2 = softmax(data1, data2)
    else:
        prob1, prob2 = softmax(probs[0:len1], probs[len1:(len1+len2)])
    # iterate data set to add weighted links
    for i in range(len1):
        d2_index = np.random.choice(range(len1, len1 + len2), int(np.ceil(prob1[i] * len2)), replace=False)
        for idx in d2_index:
            dg.add_weighted_edges_from([(idx, i, data2[idx - len1])])

    for k in range(len1, len1 + len2):
        d1_index = np.random.choice(range(len1), int(np.ceil(prob2[k - len1] * len1)), replace=False)
        for idx in d1_index:
            dg.add_weighted_edges_from([(idx, k, data1[idx])])

    # nx.draw_networkx(dg, with_labels=True)
    plt.figure()
    plt.hist(dg.degree().values())
    return dg


def iterate(data1, data2, dg, max_iter=20):
    time = 0
    while time < max_iter:
        time += 1
        probs = np.array(nx.pagerank(dg).values())
        dg = link_assign(data1, data2, probs=probs, first=False)

    return dg


men = score_generate(10, mean=65, std=10)
# plt.hist(men)

women = score_generate(10, mean=65, std=10)
# plt.hist(women)

dg = link_assign(men, women, first=True)

dgg = iterate(men, women, dg, max_iter=20)
plt.hist(dgg.degree().values())

nx.draw_networkx(dgg, with_labels=True)

print sorted(nx.degree(dgg))
for a, b in dgg.adjacency_iter():
    print a, b

dgg = iterate(men, women, dgg, max_iter=20)

men = score_generate(100, mean=65, std=10)
women = score_generate(100, mean=65, std=10)

dg = link_assign(men, women, first=True)
print nx.degree(dg)


