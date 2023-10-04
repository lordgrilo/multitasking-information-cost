import numpy as np
from scipy.optimize import minimize, basinhopping
import networkx as nx

# def res_matrix(A, num_layers, betas, S):
#     diagS = np.zeros((len(S), len(S)))
#     for i in range(len(S)):
#         if S[i]==1:
#             diagS[i,i] = 1;
#     Ss, diagSs = {}, {}
#     Ss[0] = S;
#     diagSs[0] = diagS
#     for i in range(num_layers-1):
#         diagSs[i+1] = np.dot(np.dot(diagSs[i], A), diagS)
#         for l in range(len(S)):
#             w = np.sum(diagSs[i+1][:,l])
#             if w!=0:
#                 diagSs[i+1][:,l] = diagSs[i+1][:,l] / (betas[l] + np.sum(diagSs[i+1][:,l]));
#     return diagSs;


# def objective_fun(x):
#     test_res = compute_single_task_interference(A, task, num_layers, x)
#     single_task_prob = test_res[task[0], task[-1]]
#     if single_task_prob!=0:
#         return -np.log(single_task_prob)
#     else:
#         return 0;

# def compute_single_task_interference(A, t, nl, b):
#     activated_nodes = np.zeros(A.shape[0],)
#     for s in t:
#         activated_nodes[s] = 1
#     iterated_matrices = res_matrix(A, nl, b, activated_nodes)
#     return iterated_matrices[nl-1];

# def objective_fun_multitask(x):

#     test_res = compute_multitask_interference(A, tasks, num_layers, x)
#     compound_prob = 1;
#     for task in tasks:
#         compound_prob *= test_res[task[0], task[-1]]
#     if compound_prob!=0:
#         return -np.log(compound_prob)
#     else:
#         return 0;

# def compute_multitask_interference(A, tl, nl, b):
#     activated_nodes = np.zeros(A.shape[0],)
#     for t in tl:
#         for s in t:
#             activated_nodes[s] = 1
#     iterated_matrices = res_matrix(A, nl, b, activated_nodes)
#     return iterated_matrices[nl-1];

# def activated_nodes(A, tl):
#     activated_nodes = np.zeros(A.shape[0],)
#     for t in tl:
#         for s in t:
#             activated_nodes[s] = 1
#     return activated_nodes;

# def delta_cost(mp):
#     return [(x[0]-x[1])/x[0] for x in mp];

# def delta_beta_ratio(x, betas0):
#     return np.mean((betas0 - x) / betas0);


def are_structural_interfering(p1, p2):
    if set(p1[1:]).intersection(set(p2[1:])):
        return True
    else:
        return False


def multitask_subgraph(g, paths):
    nodes = []
    for t in paths:
        nodes.extend(t)
    nodes = list(set(nodes))
    return nx.subgraph(g, nodes)


def are_functionally_interfering(g, paths):  # beware it works only for two tasks
    if are_structural_interfering(*paths):
        return False
    ms = multitask_subgraph(nx.Graph(g), paths)
    if nx.number_connected_components(ms) == len(paths):
        return False
    else:
        return True


# def weighted_task_path_interference(g, p1, p2):
#     if len(set(p1[1:]).intersection(set(p2[1:])))>0:
#         return True;
#     count = 0;
#     for l in range(len(p1)-1):
#         if p2[l+1] in list(g.successors(p1[l])) or p1[l+1] in list(g.successors(p2[l])):
#             count+=1;
#     return count;
# def raw_delta_cost(mp, normalized=False):
#     if normalized==False:
#         return [(x[0]-x[1]) for x in mp];
#     else:
#         return [(x[0]-x[1])/x[0]  if x[0]>0 else 0 for x in mp];

# def raw_delta_beta(x, betas0, func=np.mean, normalized=False):
#     if normalized==True:
#         return func((betas0 - x) / betas0);
#     else:
#         return func((betas0 - x));

# def boostrap_lr(x, y, iterations=500):
#     X, Y = np.array(x), np.array(y)
#     coeff, inter = [], []
#     for i in range(0, iterations):
#         sample_index = np.random.choice(range(0, len(y)), len(y))

#         X_samples = X[sample_index]
#         y_samples = y[sample_index]

#         lr = LinearRegression()
#         lr.fit(X_samples, y_samples)
#         coeff.append(lr.coef_)
#         inter.append(lr.intercept_)
#     return coeff, inter;

#######################################


def single_task_in_multitask_prob(
    t, sg, betas_t, nus, weight_func=np.exp, attr="weight", verbose=False
):
    prob = 1
    w = nx.get_edge_attributes(sg, attr)
    for i, n in enumerate(t[1:]):
        # probability of propagating the right path
        in_n = list(sg.predecessors(n))
        predecessor_probs = []
        Z = betas_t[n] + np.sum([weight_func(w[(x, n)] ** nus[n]) for x in in_n])
        prob *= weight_func(w[(t[i], n)] ** nus[n]) / Z
    return prob


def compute_multitask_interference(gg, tl, nl, b, nu, attr="weight"):
    probs = []
    for t in tl:
        probs.append(
            single_task_in_multitask_prob(t, gg, b, nu, weight_func=np.exp, attr=attr)
        )
    return probs


def objective_fun_multitask(x):
    probs = compute_multitask_interference(sg, new_tasks, num_layers, beta_vec, x)
    compound_prob = 1
    for prob in probs:
        compound_prob *= prob
    if compound_prob != 0:
        return -np.log(compound_prob)
    else:
        return 0


def subgraph_relabel(gg, ts):
    nn = []
    for t in ts:
        nn.extend(t)
        nn = list(set(nn))
    sg = nx.subgraph(gg, nn)
    rel_dict = dict(zip(sg.nodes(), list(range(sg.number_of_nodes()))))
    sg = nx.relabel_nodes(sg, rel_dict)
    new_ts = []
    for t in ts:
        new_ts.append([rel_dict[x] for x in t])
    return sg, new_ts
