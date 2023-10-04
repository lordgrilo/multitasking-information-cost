

import numpy as np 
import pandas as pd
import networkx as nx
import time 
from tqdm import tqdm
import operator 
import sys 
from path_overlap import * 

from scipy.optimize import minimize, basinhopping

import sys

from stroop_functions import * 


chosen_policies_dict = {}
max_lts = {}
graph_archive = {}

output_dir = '../data/'
discount_factor = sys.argv[0] # -0.0 ####[-0.0, -.1, -.5]
T_h = sys.argv[1] # 100 #### [20, 50, 500] #
num_graph_iterations = sys.argv[2]

num_task_iterations = sys.argv[3]


    #
for graph_iteration in num_graph_iterations:

    nodes = [r'$S_1^r$',r'$S_1^g$', r'$S_2^r$', r'$S_2^g$', r'$H_1^r$', r'$H_1^g$', r'$H_2^r$', r'$H_2^g$', r'$R^r$', r'$R^g$']

    beta = 10
    betas  = [0, 0, 0, 0, 
              beta, beta, beta, beta, 
             beta, beta]


    pos = [(0, 0), (0, 1), (0, 3), (0, 4),
          (1, 0), (1, 1), (1, 3), (1, 4),
          (2, 1.5), (2, 2.5)]

    betas = dict(zip(nodes,betas))
    pos = dict(zip(nodes,pos))


    w_strong = 6
    w_medium = 2
    w_weak = 1.5 


    G2 = nx.DiGraph()
    G2.add_nodes_from(nodes)
    nx.set_node_attributes(G2, betas, 'beta')
    factor = 1.8

    #input-representation links
    G2.add_edge(r'$S_1^r$',r'$H_1^r$', weight=w_medium + eps_noise()) ### mediuim
    G2.add_edge(r'$S_1^g$',r'$H_1^g$', weight=w_medium + eps_noise())
    G2.add_edge(r'$S_2^r$',r'$H_2^r$', weight=factor*w_strong + eps_noise())
    G2.add_edge(r'$S_2^g$',r'$H_2^g$', weight=factor*w_strong + eps_noise())

    # #input cross links
    G2.add_edge(r'$S_1^r$',r'$H_1^g$', weight=-w_weak + eps_noise())
    G2.add_edge(r'$S_1^g$',r'$H_1^r$', weight=-w_weak + eps_noise())
    G2.add_edge(r'$S_2^r$',r'$H_2^g$', weight=-factor*w_weak + eps_noise())
    G2.add_edge(r'$S_2^g$',r'$H_2^r$', weight=-factor*w_weak + eps_noise())

    #representation-output links
    #color naming
    G2.add_edge(r'$H_1^r$', r'$R^r$', weight=w_medium + eps_noise())
    G2.add_edge(r'$H_1^g$', r'$R^g$', weight=w_medium + eps_noise())
    G2.add_edge(r'$H_1^r$', r'$R^g$', weight=-w_weak + eps_noise())
    G2.add_edge(r'$H_1^g$','$R^r$', weight=-w_weak + eps_noise())
    #word reading
    G2.add_edge(r'$H_2^r$', r'$R^r$', weight=factor*w_strong + eps_noise())
    G2.add_edge(r'$H_2^g$', r'$R^g$', weight=factor*w_strong + eps_noise())
    G2.add_edge(r'$H_2^r$', r'$R^g$', weight=-factor*w_medium + eps_noise())
    G2.add_edge(r'$H_2^g$', r'$R^r$', weight=-factor*w_medium + eps_noise())
    edges = G2.edges()
    weights = [G2[u][v]['weight'] for u,v in edges]


    results_collection = {}


    G = nx.DiGraph()

    G.add_edge('S1', 'H1', weight = w_strong + eps_noise())
    G.add_edge('H1', 'R1', weight = w_strong + eps_noise())

    G.add_edge('S2', 'H2', weight = factor*w_strong + eps_noise())
    G.add_edge('H2', 'R1', weight = w_strong + eps_noise())
    G.add_edge('H2', 'R2', weight = factor*w_strong + eps_noise())

    G.add_edge('S2', 'H3', weight = w_weak + eps_noise())
    G.add_edge('H3', 'R2', weight = w_weak + eps_noise())



    di_pos_coords = [(0, 0), (0, 1), (0, 2), 
              (1, 0), (1, 1), (1, 2), 
              (2, 1)]

    di_pos = dict(zip(G.nodes(), di_pos_coords))



    G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(G.number_of_nodes()))))



    beta, nu0 = 1.49, 1


    classic_paths = [
        [0, 1, 2],
        [3, 4, 5],
    ]

    new_path = [3,6,5]
    classic_nodes = list(classic_paths[0])
    classic_nodes.extend(classic_paths[1])





    G_t = update_path_weight(G, factor, new_path)





    possible_covers = [
        [[1], [0, 2]],
        [[2], [0, 1]],
    ]


    restricted_covers = [
        [[1], [0, 2]],
        [[2], [0, 1]],
        [[0], [1]],
        [[1], [0]]
    ]


    beta, nu0 = 30, 1
    min_beta = 0 * beta
    num_layers = 3
    perc = .2
    beta_bounds_i = (min_beta, beta+0.001)
    nu_bounds_i = (nu0-perc, nu0+perc)
    # mrt_const = 0.1
    mrt_value = .1
    non_lin_value = 1
    rounding = 50
    learning_rate = 0.0002



    from collections import defaultdict



    tag = -1

    all_paths = list(classic_paths)
    all_paths.append(new_path) 
    actual_tasks = all_paths
    full_record = defaultdict(dict)

    sgs = []

    tracking = {}
    for t in tqdm(range(1, 2500)):
        factor = learning_rate * t;
        
        G_t = update_path_weight(G, factor, new_path)

        time_covers = {}
        serialization_possibilities = {}
        sols = {}
        for cover in restricted_covers:
            rrs = []
            sols[str(cover)] = {}
            for subset in cover[1:]:
                tasks = [actual_tasks[c] for c in subset]
                print(cover, subset, tasks)
                sg, new_tasks, rl  = subgraph_relabel(G_t, tasks, True)
                sgs.append([subset, sg, rl]);
                inv_rl = dict(zip(rl.values(), rl.keys()))
                beta_vec = [beta] * sg.number_of_nodes()
                nu_vec = [nu0] * sg.number_of_nodes()

                x = beta_vec.copy()
                bnds = [beta_bounds_i] * len(beta_vec)
                x.extend(nu_vec)
                bnds.extend([nu_bounds_i] * len(nu_vec))
                bnds = tuple(bnds)

                from scipy.optimize import NonlinearConstraint

                sol = minimize(
                    objective_fun_multitask, x, method="SLSQP", bounds=bnds
                )  # , constraints=nlc)#SLSQP
                probs = compute_multitask_interference(
                    sg,
                    list(new_tasks),
                    num_layers,
                    sol.x[: sg.number_of_nodes()],
                    sol.x[sg.number_of_nodes() :],
                )
                compound_prob = 1
                for prob in probs:
                    compound_prob *= prob
                rrs.append(compound_prob)
                sols[str(cover)][str(subset)] = sol.x
                
            time_covers[str(cover)] = local_full_rr(rrs, cover, mrt_const=mrt_value, nonlin=non_lin_value, norm_factor=.5)
            serialization_possibilities[str(cover)] = local_full_rr(
                rrs, cover, mrt_const=mrt_value, nonlin=non_lin_value, norm_factor=1
            ) 
            full_record[t][str(cover)] = time_covers[str(cover)]
            print(str(cover), time_covers[str(cover)])

        max_key = max(serialization_possibilities.items(), key=operator.itemgetter(1))[
            0
        ]
        tracking[t] = [max_key, [np.round(x, rounding) for x in time_covers[str(max_key)]]] 
        
        print('factor:', np.round(factor, rounding), 'cover:', str(max_key), [np.round(x, rounding) for x in time_covers[str(max_key)]])
        print('beta sols:', sols[max_key])
        print('\n')




    good_ks = ['Serial', 'original pathway', 'new pathway']
    ks  = ['[[1], [0]]', '[[2], [0, 1]]',  '[[1], [0, 2]]']

    good_ks = ['Serial', 'new pathway']
    ks  = ['[[1], [0]]', '[[1], [0, 2]]']



    lt0 = 1

    chosen_policies_dict[graph_iteration] = []
    max_lts[graph_iteration] = []
    for iteration in range(num_task_iterations):
        chosen_policies = []
        reward_tracking = [0]
        spent_time = [0]
        lt = lt0
        while spent_time[-1]<T_h:
            probs = policy_probs(T_h, lt, spent_time[-1], full_record, policies=ks, lambda_= discount_factor)
            chosen_policies.append(instant_policy(probs))
            ar, st = accrue_reward_spent_time(full_record, chosen_policies[-1], lt)
            reward_tracking.append(reward_tracking[-1] + ar);
            spent_time.append(spent_time[-1] + st)
            if chosen_policies[-1] == '[[1], [0, 2]]':
                lt+=20;
        max_lts[graph_iteration].append(lt);
        chosen_policies_dict[graph_iteration].extend(chosen_policies)


pk.dump(chosen_policies_dict, 
    open(output_dir + 'output_'+str(discount_factor)+'_'+str(T_h))+'.pck', 'wb');






