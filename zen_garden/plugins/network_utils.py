"""
Shared network helpers for ZEN-garden plugins.

ZEN-garden represents every corridor as two directed edges, and several plugins
need to find the edge running the other way: DC power flow writes Kirchhoff's
voltage law on a directed pair, and target constraints net the two directions
against each other.

Pairing on the node pair alone is only safe when at most one corridor connects
any two nodes. Aggregated networks routinely break that: exporting a clustered
grid without merging parallel corridors leaves several branches between the same
pair of aggregated buses, each carrying its own rating and impedance. Keying a
lookup on the node pair then silently keeps one of them and pairs every parallel
branch with the same partner.

Exporters that emit one edge per physical asset encode the asset in the edge id,
conventionally ``<from>-<to>__<asset>``. Where that suffix is present it makes
the partner unambiguous; where it is absent the node pair is used, and an
ambiguous pair is reported rather than resolved arbitrarily.
"""

ASSET_SEPARATOR = "__"


def _pair_key(edge, node_from, node_to):
    """Build the identity of a directed edge for partner lookup.

    :param edge: edge id
    :param node_from: origin node
    :param node_to: destination node
    :return: tuple identifying the directed edge
    """
    asset = (
        edge.split(ASSET_SEPARATOR, 1)[1] if ASSET_SEPARATOR in edge else None
    )
    return node_from, node_to, asset


def build_reverse_edge_map(nodes_on_edges):
    """Map every directed edge to the edge running the other way.

    :param nodes_on_edges: dict edge -> (node_from, node_to)
    :return: (reverse_by_edge, ambiguous_edges) where reverse_by_edge maps an
        edge to its partner or to None when no unambiguous partner exists, and
        ambiguous_edges is the set of edges that share a node pair with another
        edge without an asset suffix to tell them apart
    """
    edge_by_key = {}
    duplicate_keys = set()
    for edge, (node_from, node_to) in nodes_on_edges.items():
        key = _pair_key(edge, node_from, node_to)
        if key in edge_by_key:
            duplicate_keys.add(key)
        edge_by_key[key] = edge

    reverse_by_edge = {}
    ambiguous_edges = set()
    for edge, (node_from, node_to) in nodes_on_edges.items():
        key = _pair_key(edge, node_from, node_to)
        reverse_key = (key[1], key[0], key[2])
        if key in duplicate_keys or reverse_key in duplicate_keys:
            ambiguous_edges.add(edge)
            reverse_by_edge[edge] = None
        else:
            reverse_by_edge[edge] = edge_by_key.get(reverse_key)
    return reverse_by_edge, ambiguous_edges


def describe_ambiguous_edges(ambiguous_edges, nodes_on_edges, limit=10):
    """Render ambiguous edges as a message fragment for an error or warning.

    :param ambiguous_edges: edges with no unambiguous reverse partner
    :param nodes_on_edges: dict edge -> (node_from, node_to)
    :param limit: how many edges to list before truncating
    :return: human-readable description
    """
    listed = sorted(ambiguous_edges)
    shown = ", ".join(
        f"{e} ({nodes_on_edges[e][0]}->{nodes_on_edges[e][1]})" for e in listed[:limit]
    )
    suffix = f" ... and {len(listed) - limit} more" if len(listed) > limit else ""
    return (
        f"{len(listed)} edge(s) share a node pair with no '{ASSET_SEPARATOR}<asset>' "
        f"suffix to distinguish them, so their reverse partner is undefined: "
        f"{shown}{suffix}"
    )
