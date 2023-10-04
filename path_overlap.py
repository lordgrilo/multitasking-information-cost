
import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time 
from tqdm.notebook import tqdm
from itertools import combinations, product

def edges_to_next_layer(origin_node, target_nodes, prob):
    return [(origin_node,x) for x in target_nodes if np.random.rand()<=prob];

    

def multipartite_network(num_layers, num_dense_units, num_sparse_units, density = 1, capacity=True):
    # the full network will have num_layer * (num_dense_units + num_sparse_units) nodes.
    num_tot_units = num_dense_units + num_sparse_units
    g = nx.DiGraph();
    multipartite_dict = {}
    layer_nodes = {}
    for l in range(num_layers):
        layer_nodes[l] = [(l, x) for x in range(num_tot_units)];
        g.add_nodes_from(layer_nodes[l]);
        for node in layer_nodes[l]:
            multipartite_dict[node] = node[0];
    nx.set_node_attributes(g, multipartite_dict, 'subset')
    ### add links in the dense part of the network
    for l in range(num_layers-1):
        for n in range(num_dense_units):
            g.add_edges_from(edges_to_next_layer((l,n), layer_nodes[l+1][:num_dense_units], density))
    ### add links in the dense part of the network + links between "same" units in different layers
    for l in range(num_layers-1):
        for n in range(0, num_tot_units):
            g.add_edge((l,n), (l+1, n));
    if capacity==True:
        edge_dict = dict(zip(list(g.edges()), [1]*g.number_of_edges()));
        nx.set_edge_attributes(g, edge_dict,'capacity');
       	node_dict = dict(zip(list(g.nodes()), [1]*g.number_of_nodes()));
        nx.set_node_attributes(g, node_dict,'capacity');
    return g;


def rewire_multipartite_network(g, num_tot_units, rewire_prob=.2, rewire_iters=100, verbose=False, capacity=True):
    rew_g = g.copy()
    num_edges = rew_g.number_of_edges()
    rewire_count = 0
    for it in range(rewire_iters):
        candidate_edge = list(rew_g.edges())[np.random.randint(0, num_edges)]

        if rew_g.in_degree(candidate_edge[1]) > 1:  # we want to preserve nodes
            if np.random.rand() <= rewire_prob:

                candidate_nodes = [(candidate_edge[1][0], x) for x in range(
                    num_tot_units) if x != candidate_edge[0][1]]

                new_node = candidate_nodes[np.random.randint(
                    0, len(candidate_nodes))]

                new_edge = [candidate_edge[0], new_node]

                if not rew_g.has_edge(new_edge[0], new_edge[1]):
                    rew_g.remove_edge(candidate_edge[0], candidate_edge[1])
                    if capacity==True:
                    	rew_g.add_edge(new_edge[0], new_edge[1], capacity=1)
                    else:
                    	rew_g.add_edge(new_edge[0], new_edge[1], capacity=1)
                    rewire_count += 1
    if verbose == True:
        return rew_g, rewire_count
    else:
        return rew_g
    
    
    
def check_task_path_interference(path1, path2):
    p1 = list(map(str, path1))
    p2 = list(map(str, path2))
    if np.any(np.array(p1) == np.array(p2)):
        return True; #paths interfere
    else:
        return False; #paths do not interfere
    
    
    
def interference_overlap(ps1, ps2):
    from itertools import product

    if len(ps1) * len(ps2) == 0:
        return None;

    interference_count = 0    
    for p1, p2 in product(ps1, ps2):
            if check_task_path_interference(p1, p2):
                interference_count += 1;
    return interference_count / (len(ps1) * len(ps2));


def collect_task_path_nodes(ps):
    nodes = []
    for p in ps:
        nodes.extend(p)
    return set(nodes);

def node_interference_overlap(ps1, ps2):
    from itertools import product

    if len(ps1) * len(ps2) == 0:
        return None;
    nodes1, nodes2 = collect_task_path_nodes(ps1), collect_task_path_nodes(ps2)
    return len(nodes1.intersection(nodes2)) / len(nodes1.union(nodes2))


def check_layer_level_interference(g, p1, p2):
    if len(set(p1).intersection(set(p2)))>0:
        return True;
    for l in range(len(p1)-1):
        if p2[l+1] in list(g.successors(p1[l])) or p1[l+1] in list(g.successors(p2[l])):
            return True;
    return False;


def weighted_task_path_interference(g, p1, p2):
    if len(set(p1).intersection(set(p2)))>0:
        return True;
    count = 0;
    for l in range(len(p1)-1):
        if p2[l+1] in list(g.successors(p1[l])) or p1[l+1] in list(g.successors(p2[l])):
        	count+=1;
    return count;

def layer_interference_overlap(g, ps1, ps2):
    from itertools import product

    if len(ps1) * len(ps2) == 0:
        return None;
    interference_count = 0    
    for p1, p2 in product(ps1, ps2):
            if check_layer_level_interference(g, p1, p2):
                interference_count += 1;
    return interference_count / (len(ps1) * len(ps2));


def draw_individual_task_path(G, pos, path, axis, color, width=1.0):
    path_edges = []
    for i in range(len(path)-1):
        path_edges.append([path[i], path[i+1]])
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=path_edges,
        width=8,
        alpha=0.5,
        edge_color=color,
        ax=axis,
    )
    return;

def draw_task_path_family(G, pos, paths, axis, color, width=1.0):
    path_edges = []
    for path in paths:
        for i in range(len(path)-1):
            path_edges.append([path[i], path[i+1]])
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=path_edges,
        width=8,
        alpha=0.5,
        edge_color=color,
        ax=axis,
    )
    return;

 
def draw_task_path_shadow(g, pos, path, axis, color, width=8.0, alpha_frac=0.5):
    shadow = []
    shadow_edges = []
    for node in path[:-1]:
        next_nodes = list(g.successors(node));
        shadow.extend(next_nodes);
        for n in next_nodes:
            shadow_edges.append([node, n]);
    shadow = list(set(shadow))
    
    nx.draw_networkx_nodes(g,pos,
                       nodelist=shadow,
                       node_color=color,
                       node_size=200,
                   alpha=alpha_frac)
    
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=shadow_edges,
        width=width,

        alpha=alpha_frac,
        edge_color=color,
        ax=axis,
    )
    return;



def draw_individual_task_path(g, pos, path, axis, color, width=8.0, alpha_frac=0.8):
    path_edges = []
    for i in range(len(path)-1):
        path_edges.append([path[i], path[i+1]])
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=path_edges,
        width=width,
        alpha=alpha_frac,
        edge_color=color,
        ax=axis,
    )
    return;


def MP_interference_graph(graph, paths=None):
    return_paths = False;
    if paths==None:
        return_paths = True;
        mp_dict = nx.get_node_attributes(graph, 'subset');
        tot_units = np.sum(np.array(list(nx.get_node_attributes(graph, 'subset').values()))==0);
        vals = list(set(list(mp_dict.values())))
        min_set, max_set = np.min(vals), np.max(vals);
        output_nodes = [(max_set, x) for x in range(tot_units)]
        input_nodes = [(min_set, x) for x in range(tot_units)]
        paths = {}
        for n, nn in product(input_nodes, output_nodes):
            ps = list(nx.all_simple_paths(graph, n, nn));
            if len(ps)>0:
                paths[(n, nn)] = ps;
    ig = nx.Graph() # interference graph
    ig.add_nodes_from(list(paths.keys()));
    for tf1, tf2 in combinations(paths.keys(),2):
        w = layer_interference_overlap(graph, paths[tf1], paths[tf2]);
        if w!=None and w>0:
            ig.add_edge(tf1, tf2, weight=w);
    if return_paths==True:
        return ig, paths;
    else:
        return ig;



def MP_task_interference_graph(graph, paths=None):
    return_paths = False;
    if paths==None:
        return_paths = True;
        mp_dict = nx.get_node_attributes(graph, 'subset');
        tot_units = np.sum(np.array(list(nx.get_node_attributes(graph, 'subset').values()))==0);
        vals = list(set(list(mp_dict.values())))
        min_set, max_set = np.min(vals), np.max(vals);
        output_nodes = [(max_set, x) for x in range(tot_units)]
        input_nodes = [(min_set, x) for x in range(tot_units)]
        paths = {}
        for n, nn in product(input_nodes, output_nodes):
            ps = list(nx.all_simple_paths(graph, n, nn));
            if len(ps)>0:
                paths[(n, nn)] = ps;
    node_path_list = []
    for k in paths:
        for kk in paths[k]:
            node_path_list.append(tuple(paths[k][kk]))
    ig = nx.Graph() # interference graph
    ig.add_nodes_from(node_path_list);
    for tf1, tf2 in combinations(node_path_list,2):
        w = weighted_task_path_interference(graph, paths[tf1], paths[tf2]);
        if w>0:
            ig.add_edge(tf1, tf2, weight=w);
    if return_paths==True:
        return ig, paths;
    else:
        return ig;



def HO_interference(path_families, graph):
    interference_combos = 0
    combos = list(product(*path_families));
    norm = len(combos)
    if norm==0:
        return None;
    for paths in combos:
        for p1, p2 in combinations(paths, 2):
            if check_layer_level_interference(graph, p1, p2):
                interference_combos+=1;
                break;          
    return interference_combos / norm;


def effective_capacity(graph, graph_paths, sizes=[2,3,4], num_samples=10000):
    from tqdm.notebook import tqdm
    from random import sample
    capacity = {}
    for k in sizes:
        capacity[k] = []
        counts = 0
        for mt_combo in combinations(graph_paths.keys(), k):
            w = HO_interference([graph_paths[x] for x in mt_combo], graph);
            if w!=None:
                capacity[k].append(w);
            # else:
            #     capacity[k].append(0)
            counts+=1
            if counts>num_samples:
                break
        capacity[k] = np.mean(capacity[k])
    return capacity;


def all_flows_MPG(graph, graph_paths):
    from networkx.algorithms.flow import shortest_augmenting_path
    flow_value = {};
    for k in graph_paths.keys():
        # try:
        flow_value[k] = nx.maximum_flow_value(graph, k[0], k[1], flow_func=shortest_augmenting_path);
        # except:
        #     flow_value[k] = np.nan
    return flow_value;


def weighted_task_path_interference(g, p1, p2):
    if len(set(p1).intersection(set(p2)))>0:
        return True;
    count = 0;
    for l in range(len(p1)-1):
        if p2[l+1] in list(g.successors(p1[l])) or p1[l+1] in list(g.successors(p2[l])):
        	count+=1;
    return count;


def MP_task_interference_graph(graph, paths=None):
    return_paths = False;
    if paths==None:
        return_paths = True;
        mp_dict = nx.get_node_attributes(graph, 'subset');
        tot_units = np.sum(np.array(list(nx.get_node_attributes(graph, 'subset').values()))==0);
        vals = list(set(list(mp_dict.values())))
        min_set, max_set = np.min(vals), np.max(vals);
        output_nodes = [(max_set, x) for x in range(tot_units)]
        input_nodes = [(min_set, x) for x in range(tot_units)]
        paths = {}
        for n, nn in product(input_nodes, output_nodes):
            ps = list(nx.all_simple_paths(graph, n, nn));
            if len(ps)>0:
                paths[(n, nn)] = ps;
    node_path_list = []
    for k in paths:
        for kk in paths[k]:
            node_path_list.append(tuple(kk))
    ig = nx.Graph() # interference graph
    ig.add_nodes_from(node_path_list);
    for tf1, tf2 in combinations(node_path_list,2):
        w = weighted_task_path_interference(graph, tf1, tf2);
        if w>0:
            ig.add_edge(tf1, tf2, weight=w);
    if return_paths==True:
        return ig, paths;
    else:
        return ig;

    
    