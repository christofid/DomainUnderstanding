import networkx as nx

def reorder_labels(labels, new_order):

    new_labels = []

    for x in labels:
       new_labels.append( sorted(x,  key = lambda i: new_order.index(i)) )

    return new_labels


def find_ordering(labels, order_type, root_node):

    edges = []
    added_edges = []
    for label in labels:
        x,name,y = label.split('.')

        i=-1
        if (x,y) in added_edges:
            i = added_edges.index((x,y))
        elif (y,x) in added_edges:
            i = added_edges.index((y,x))

        if i!=-1:
            edges[i][2]['name'].append(name)
        else:
            edges.append((x,y,{'name':[name]}))
            added_edges.append((x,y))

    G = nx.Graph()
    G.add_edges_from(edges)

    if root_node=='none':
        degrees = nx.degree_centrality(G)
        root_node =  max(degrees, key=degrees.get)

    if order_type == 'bfs':
         new_labels = nx.edge_bfs(G, root_node)
    if order_type == 'dfs':
        new_labels = nx.edge_dfs(G, root_node)

    names = nx.get_edge_attributes(G, 'name')

    ordered_labels = []
    for label in new_labels:

        x,y = label

        if label in names:
            edge_names = names[label]
        else:
            edge_names = names[(y, x)]

        for name in edge_names:
            new_label = '.'.join([x,name,y])

            if new_label in labels:
                ordered_labels.append(new_label)
            else:
                new_label = '.'.join([y, name, x])
                ordered_labels.append(new_label)

    return ordered_labels
