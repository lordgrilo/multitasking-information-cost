import networkx as nx 
import numpy as np 
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt

def prob_fire(g, node, sources, verbose=False):
    if not isinstance(sources, list):
        sources = [sources]
    b = g.nodes()[node]['beta']
    edges = g.edges();
    input_weights = np.sum([edges[(x, node)]['weight'] for x in sources if x in g.predecessors(node)])
    if verbose==True:
        print(input_weights)
    if input_weights!=0:
        return np.exp(input_weights) / (b + np.exp(input_weights));
    else:
        return 0;

    
def draw_trajectory(g, traj, ppos, factor=3, all_edges=True, inactive_on=False):
    active, inactive  = traj;
    if all_edges:
        edges = g.edges()
        w = [g[u][v]['weight'] for u,v in edges]
        nx.draw_networkx_edges(g, ppos, edge_color='gray')
    
    gg = nx.subgraph(g, active)
    edges = gg.edges()
    ww = [factor * gg[u][v]['weight'] for u,v in edges]
    nx.draw(gg, pos=ppos, width=ww)

    nx.draw_networkx_nodes(g, ppos, nodelist=active, 
                           node_color='black')
    if inactive_on==True:

        nx.draw_networkx_nodes(g, ppos, nodelist=inactive, 
            node_color='white', 
            edgecolors='black',
            linewidths=2);
        
        useless = list(set(g.nodes()) - set(active) - set(inactive))
    else: 
        useless = list(set(g.nodes()) - set(active))

    nx.draw_networkx_nodes(g, ppos, nodelist=useless, 
        node_color='white', 
        edgecolors='gray', alpha=.5,
        linewidths=1);

    plt.xlim(-1,5)
    plt.ylim(-1, 3)
    return;

def traj_prob(g, traj, verbose=False):
    probs, not_probs = 1, 1
    for node in traj[0]:
        if node[1]=='S':
            continue;
        else:
            if verbose==True:
                print('+', node, prob_fire(g, node, traj[0]))
            probs *= prob_fire(g, node, traj[0])
            
        for node in traj[1]:
            if verbose==True:
                print('-', node, 1 - prob_fire(g, node, traj[0]))
            not_probs *= (1 - prob_fire(g, node, traj[0]))
    return probs, not_probs;


def build_trajectory(combo, sources, internal_nodes, correct_responses, incorrect_responses):
    active = list(sources)
    active.extend(correct_responses);
    active.extend(combo);
    inactive = list(incorrect_responses)
    inactive.extend(list(set(internal_nodes) - set(combo)))
    return [active, inactive];

    
def trajectory_set_construction(g, sources, correct_responses, incorrect_responses):
    stimulus_nodes = [x for x in g.nodes() if g.in_degree(x)==0]
    internal_nodes = set(g.nodes()) - set(sources) - set(correct_responses) - set(incorrect_responses) - set(stimulus_nodes);
    internal_nodes = list(internal_nodes)
    M = len(internal_nodes);
    
    tset = []
    for m in tqdm(range(1, M)):
        for combo in combinations(internal_nodes, m):
            new_traj = build_trajectory(combo, sources, internal_nodes, 
                                        correct_responses, incorrect_responses)
            tset.append(new_traj);
    return tset;
