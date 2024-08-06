import networkx as nx
import torch.nn as nn


def remove_and_rewire(G, node_id):
    source_edges, dest_edges = G.remove(node_id)
    source_ids = []
    for s, _, data in source_edges:
        if data["outlet"] == "main" and data["inlet"] == "main":
            source_ids.append(s)
    dest_ids = []
    for _, d, data in dest_edges:
        if data["outlet"] == "main" and data["inlet"] == "main":
            dest_ids.append(d)
    for s in source_ids:
        for d in dest_ids:
            G.connect(s, d)


def prune_parameters(G_tensor, graph_parameters, prune_mask, node_config):
    keep_mask = ~prune_mask
    pruned_graph_parameters = {}
    for node_type, parameters in graph_parameters.items():
        idx = node_config.node_type_to_index[node_type]
        keep_mask_type = keep_mask[G_tensor.node_types == idx]
        pruned_graph_parameters[node_type] = nn.ParameterDict(
            {k: v[keep_mask_type] for k, v in parameters.items()}
        )
    return nn.ModuleDict(pruned_graph_parameters)


def prune_grafx(G, prune_mask, types_to_keep=["in", "mix", "out"]):
    G = G.copy()
    node_list = list(G.nodes)
    for i in node_list:
        if prune_mask[i]:
            node = G.nodes[i]
            node_type = node["node_type"]
            if not node_type in types_to_keep:
                remove_and_rewire(G, i)
    node_ids = list(G.nodes())
    relabel_mapping = {node_ids[i]: i for i in range(G.number_of_nodes())}
    G = nx.relabel_nodes(G, relabel_mapping)
    G.consecutive_ids = True
    return G
