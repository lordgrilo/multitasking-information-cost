
def MRT(rr, const=1):
    return const * (1 - 2 * (1 - rr)) * np.log(rr / (1 - rr))


def MRT2(rr, cover_item, const=1, epsilon=0.01):
    return const * (np.log(1 / (1 + epsilon - rr))) + const * np.log(1 + len(cover_item)) * (
        1.37  +  0.147 * np.log(1 / (1 + epsilon - rr))
    )

def MRT3(rr, cover_item, const=1, epsilon=0.01):
    n = len(cover_item);
    return  const * (1 - (n+1)*(1-rr)/n) * np.log(n*n*rr/(1-rr)) 


# def MRT33(rr, cover_item, const=1, epsilon=0.01):
#     n = len(cover_item);
#     return  epsilon + const *  (1 - (n+1)*(1-rr)/n) * np.log(n*n*rr/(1-rr)) 


def reconf_cost(n_prev, n_next=0, nonlin=1, norm_factor=1):
    if n_next != None:
        return ((n_prev * n_next) ** nonlin) / norm_factor
    else:
        return 1 / norm_factor


def tot_reconf_cost(cover, nonlin=0.5, norm_factor=1, log=True):
    accrued_cost = 0
    for step in range(len(cover)):
        if step == len(cover) - 1:
            accrued_cost += reconf_cost(len(cover[step]), nonlin, norm_factor)
        else:
            accrued_cost += reconf_cost(
                len(cover[step]), len(cover[step + 1]), nonlin, norm_factor
            )
    if log == True:
        return np.log(accrued_cost)
    else:
        return accrued_cost


def new_strong_weak_graph(
    num_layers,
    num_dense,
    num_sparse,
    weak_density,
    strong_mu,
    strong_std,
    weak_mu,
    weak_std,
    frac=0.8,
):
    tot_units = num_dense + num_sparse
    G = multipartite_network(num_layers, num_dense, num_sparse, 0.0)
    rel_dict = dict(zip(G.nodes(), range(G.number_of_nodes())))
    inv_rel_dict = dict(zip(range(G.number_of_nodes()), G.nodes()))

    G = nx.relabel_nodes(G, rel_dict)
    strong_w = np.random.normal(strong_mu, strong_std, G.number_of_edges())
    nx.set_edge_attributes(G, dict(zip(G.edges(), strong_w)), "weight")
    strong_paths = []
    for s in range(tot_units):
        for t in range((num_layers - 1) * tot_units, (num_layers) * tot_units):
            strong_paths.extend(nx.all_simple_paths(G, s, t))

    Gr = multipartite_network(num_layers, num_dense + num_sparse, 0, weak_density)
    Gr = rewire_multipartite_network(
        Gr, tot_units, 1, rewire_iters=int(frac * Gr.number_of_edges())
    )
    Gr = nx.relabel_nodes(Gr, rel_dict)
    weak_mu, weak_std = 1, 0.2
    weak_w = np.random.normal(weak_mu, weak_std, Gr.number_of_edges())
    weak_w_dict = dict(zip(Gr.edges(), weak_w))
    for edge in Gr.edges():
        if not G.has_edge(edge[0], edge[1]):
            G.add_edge(edge[0], edge[1], weight=weak_w_dict[edge])
    del Gr
    weak_paths = []
    for s in range(tot_units):
        for t in range((num_layers - 1) * tot_units, (num_layers) * tot_units):
            weak_paths.extend(nx.all_simple_paths(G, s, t))

    for path in strong_paths:
        if path in weak_paths:
            weak_paths.remove(path)
    return G, strong_paths, weak_paths


def subgraph_relabel(gg, ts, dic=False):
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
    if dic == False:
        return sg, new_ts
    else:
        return sg, new_ts, rel_dict


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

def single_task_in_multitask_prob(
    t, sg, betas_t, nus, weight_func=np.exp, attr="capacity", verbose=False
):
    prob = 1
    w = nx.get_edge_attributes(sg, attr)
    for i, n in enumerate(t[1:]):
        # probability of propagating the right path
        in_n = list(sg.predecessors(n))
        predecessor_probs = []
        Z = weight_func(betas_t[n]) + np.sum(
            [
                weight_func(w[(x, n)] ** nus[n])
                for x in in_n
                if sg.in_degree(x) > 0 or x == t[0]
            ]
        )
        prob *= weight_func(w[(t[i], n)] ** nus[n]) / Z
    return prob


def compute_multitask_interference(gg, tl, nl, b, nu):
    probs = []
    for t in tl:
        probs.append(
            single_task_in_multitask_prob(
                t, gg, b, nu, weight_func=np.exp, attr="weight"
            )
        )
    return probs


def objective_fun_multitask(x):
    n = int(len(x) / 2)
    probs = compute_multitask_interference(sg, new_tasks, num_layers, x[:n], x[n:])
    compound_prob = 1
    for prob in probs:
        compound_prob *= prob
    if compound_prob != 0:
        return -np.log(compound_prob)
    else:
        return 0


def full_rr(rrs, cover, mrt_const=1, nonlin=1, norm_factor=1):
    tot_cost = 1
    s = 0
    for rr in rrs:
        tot_cost *= rr
        s += MRT(rr, mrt_const)
    rc = tot_reconf_cost(cover, nonlin, norm_factor)
    return tot_cost / (s + rc)


def exploded_full_rr(rrs, cover, mrt_const=1, nonlin=1, norm_factor=1):
    tot_cost = 1
    s = 0
    for rr in rrs:
        tot_cost *= rr
        s += MRT(rr, mrt_const)
    rc = tot_reconf_cost(cover, nonlin, norm_factor)
    return tot_cost, s, rc


def report_interference_patterns(gg, ts):
    rep = []
    for comb in combinations(range(3), 2):
        rep.append(
            [
                comb,
                are_structural_interfering(*[ts[x] for x in comb]),
                are_functionally_interfering(gg, [ts[x] for x in comb]),
            ]
        )
    return rep


# In[35]:


def instant_weight_update(omega_0, t, t0=1, func=np.log, factor=1):
    return omega_0 * (1.0 + factor * float(func(t/t0)));


# In[36]:


def eps_noise(scale=0.5):
    return np.random.random() * scale;


# In[37]:


def update_edge_weights(gt, g0, path, t, func=np.log, factor=1):
    for i in range(len(path)-1):
        gt[path[i]][path[i + 1]]["weight"] = instant_weight_update(
            g0[path[i]][path[i + 1]]["weight"], t, func=func, factor=factor
        )
    return gt;


# In[38]:





def MRT33(rr, cover_item, const=.4, epsilon=0.001): #changed from const = 1 
    n = len(cover_item);
    return  epsilon + const *  (1 - (n+1)*(1-rr)/n) * np.log(n*n*rr/(1-rr)) 





def update_path_weight(g0, factor, path):
    gt = nx.DiGraph(g0);
    for i in range(len(path)-1):
        gt[path[i]][path[i + 1]]["weight"] = factor*g0[path[i]][path[i + 1]]["weight"]
    return gt;



def local_full_rr(rrs, cover,  mrt_const=1, nonlin=1, norm_factor=1):
    cv = [cover[1]]
    tot_cost = 1;
    s = 0;
    for i, rr in enumerate(rrs):
        tot_cost *= rr;
        s += MRT33(rr, cover[i])
#         s += MRT3(rr, cover[i], mrt_const);
#         s += MRT(rr, mrt_const);
    rc = tot_reconf_cost(cv, nonlin, norm_factor, log=False);

    return [np.round(x,3) for x in [tot_cost/s, tot_cost, s, rc, tot_cost/(s+rc)]];



def projected_reward(policy, T_h, it, t, ft, lambda_=0):
    itt = np.min([it, np.max(list(ft.keys()))-1])
    rt = ft[itt][policy][-1]
    R = rt * (T_h - t) * np.exp(lambda_ * (T_h - t))
    return R; 

def policy_probs(T_h, it, t, ft, lambda_=0, policies=None, epsilon = 0.0001):
    if policies==None:
        policies = ['[[1], [0, 2]]', '[[1], [0]]', '[[2], [0, 1]]'];
    Rs = [epsilon + projected_reward(x, T_h, it, t, ft, lambda_) for x in policies];
    totR = np.sum(Rs);
    return dict(zip(policies, [x/totR for x in Rs]));

def instant_policy(pp):
    k, w = list(pp.keys()), list(pp.values());
    policy_id = np.random.choice(k, 1, p=w)
    return policy_id[0];

def accrue_reward_spent_time(ft, policy, it):
    itt = np.min([it, np.max(list(ft.keys()))-1])
    return ft[itt][policy][1], ft[itt][policy][2]; # 1 is the entry for pure reward 