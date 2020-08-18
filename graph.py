import numpy as np

from btree.table import Index, Schema, Table
from einsum.einsum import einsum
from tensor.sparse import SparseTensor


class Graph(object):
    def __init__(self, node_schema, edge_schema):
        self.nodes = {}
        for node_type, (node_key, node_values) in node_schema.items():
            schema = Schema(node_key, node_values)
            self.nodes[node_type] = Table(Index(schema))

        self._node_indices = Table(
            Index(Schema([("type", str), ("key", tuple)], [("index", int)])))
        self._node_indices.add_index("node_index", ("index",))

        self._edges = {}
        for edge_label, edge_dtype in edge_schema.items():
            self._edges[edge_label] = SparseTensor([0, 0], edge_dtype)

        self._max_id = 0

    def __eq__(self, other):
        if set(self.nodes.keys()) != set(other.nodes.keys()):
            return False

        if set(self._edges.keys()) != set(other._edges.keys()):
            return False

        for node_type in self.nodes.keys():
            if self.nodes[node_type].schema() != other.nodes[node_type].schema():
                return False
            if self.nodes[node_type].count() != other.nodes[node_type].count():
                return False

        for label in self._edges.keys():
            this = self._edges[label]
            that = other._edges[label]

            if this.dtype != that.dtype:
                return False

            if this.filled_count() != that.filled_count():
                return False

        these_nodes, these_edges = self.as_stream()
        those_nodes, those_edges = other.as_stream()

        for this, that in zip(these_nodes.values(), those_nodes.values()):
            this, that = iter(this), iter(that)
            for this_node, that_node in zip(this, that):
                if this_node != that_node:
                    return False

        for this, that in zip(these_edges.values(), those_edges.values()):
            for this_edge, that_edge in zip(this, that):
                if this_edge != that_edge:
                    return False


        return True

    def add_node(self, node_type, key, value=tuple()):
        self.nodes[node_type].insert(key + value)

        self._node_indices.insert([node_type, key, self._max_id])
        self._max_id += 1

        for edges in self._edges.values():
            edges.expand([self._max_id, self._max_id])

    def add_edge(self, label, from_node, to_node, value=True):
        from_node_index = self._get_node_index(from_node)
        to_node_index = self._get_node_index(to_node)
        self._edges[label][(from_node_index, to_node_index)] = value

    def as_stream(self):
        nodes = {
            node_type: nodes.slice({})
            for node_type, nodes in self.nodes.items()
        }

        edges = {
            label: self._stream_edges(label)
            for label in self._edges.keys()
        }

        return (nodes, edges)

    def bft(self, edge_label, node, limit=None):  # breadth-first traversal
        edges = self._edges[edge_label]

        node_index = self._get_node_index(node)

        visited = SparseTensor([self._max_id], np.bool)

        adjacent = SparseTensor([self._max_id], np.bool)
        adjacent[node_index] = True

        i = 0
        while adjacent.any():
            visited = visited | adjacent
            adjacent = einsum('ji,j->i', edges, adjacent).copy()
            adjacent.mask(visited)

            for (i,), _ in adjacent.filled():
                yield self._get_node(i)

            i += 1
            if limit and i >= limit:
                break

    def _get_node(self, node_index):
        [node] = list(
            self._node_indices.slice({"index": node_index})
            .select(("type", "key")))
        return node

    def _get_node_index(self, node):
        (node_type, node_key) = node
        [(node_index,)] = list(self._node_indices.slice({
            "type": node_type, "key": node_key}).select(["index"]))
        return node_index

    def _stream_edges(self, label):
        for (from_id, to_id), weight in self._edges[label].filled():
            from_node = self._get_node(from_id)
            to_node = self._get_node(to_id)
            yield (from_node, to_node, weight)


def test_bft():
    g = Graph({"node": ([("key", int)], [])}, {"edge": np.bool})

    key = lambda i: (i,)

    g.add_node("node", key(1))
    g.add_node("node", key(2))
    g.add_node("node", key(3))
    g.add_node("node", key(4))
    g.add_node("node", key(5))
    g.add_node("node", key(6))

    g.add_edge("edge", ("node", key(1)), ("node", key(2)))
    g.add_edge("edge", ("node", key(2)), ("node", key(3)))
    g.add_edge("edge", ("node", key(3)), ("node", key(4)))
    g.add_edge("edge", ("node", key(4)), ("node", key(5)))

    found = list(g.bft("edge", ("node", key(1))))
    assert found == [("node", key(i)) for i in range(2, 6)]


def test_equals():
    g1 = Graph({"node": ([("key", int)], [])}, {"edge": np.bool})
    g2 = Graph({"node": ([("key", int)], [])}, {"edge": np.bool})

    key = lambda i: (i,)

    g1.add_node("node", key(1))
    g1.add_node("node", key(2))
    g2.add_node("node", key(1))
    g2.add_node("node", key(2))

    g1.add_edge("edge", ("node", key(1)), ("node", key(2)))
    g2.add_edge("edge", ("node", key(1)), ("node", key(2)))

    assert g1 == g2

    g1.add_edge("edge", ("node", key(2)), ("node", key(1)))

    assert g1 != g2

    g2.add_edge("edge", ("node", key(2)), ("node", key(1)))

    assert g1 == g2

    g2.add_node("node", key(3))

    assert g1 != g2

    g1.add_node("node", key(3))
    g1.add_edge("edge", ("node", key(1)), ("node", key(3)))
    g2.add_edge("edge", ("node", key(2)), ("node", key(3)))

    assert g1 != g2


def test_transaction():
    g = Graph(
        {"provider": ([("name", str)], [("resolved", bool), ("value", str)])},
        {"precedes": np.bool, "requires": np.bool})

    providers = g.nodes["provider"]

    providers.add_index("resolved", ("resolved",))

    g.add_node("provider", ("one",), (False, "this",))
    g.add_node("provider", ("two",), (False, "is",))
    g.add_node("provider", ("three",), (False, "a",))
    g.add_node("provider", ("four",), (False, "transaction",))

    g.add_edge("precedes", ("provider", ("one",)), ("provider", ("two",)))
    g.add_edge("requires", ("provider", ("two",)), ("provider", ("one",)))

    g.add_edge("precedes", ("provider", ("two",)), ("provider", ("three",)))
    g.add_edge("requires", ("provider", ("three",)), ("provider", ("two",)))

    g.add_edge("precedes", ("provider", ("three",)), ("provider", ("four",)))
    g.add_edge("requires", ("provider", ("four",)), ("provider", ("three",)))

    done = False
    while not done:
        done = True
        unresolved = g.nodes["provider"].slice({"resolved": False}).select(("name",))
        for (value_id,) in unresolved:
            done = False
            ready = True

            for (_, (name,)) in g.bft("requires", ("provider", (value_id,)), 1):
                provider = providers.slice({"name": name}).select(("resolved", "value"))
                [(provider_ready, provider_value)] = list(provider)
                if not provider_ready:
                    ready = False
                    break

            if ready:
                providers.slice({"name": value_id}).update({"resolved": True})

    pass # no error = test passes

if __name__ == "__main__":
    test_bft()
    test_equals()
    test_transaction()
    print("PASS")

