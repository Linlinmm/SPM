# --------------------------------------------------------

# article{feng2018hypergraph,
#   title={Hypergraph Neural Networks},
#   author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
#   journal={AAAI 2019},
#   year={2018}
# }

# --------------------------------------------------------
import numpy as np


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[104], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H






def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]

    W = np.ones(n_edge)

    DV = np.sum(H * W, axis=1)

    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G
