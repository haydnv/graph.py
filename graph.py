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

        self.node_ids = Table(
            Index(Schema([("type", str), ("key", tuple)], [("id", int)])))
        self.node_ids.add_index("node_id", ("id",))

        self.edge_data = {}
        self.edges = {}
        for edge_type, edge_values in edge_schema.items():
            if edge_values is None:
                self.edge_data[edge_type] = None
            else:
                schema = Schema([("from", int), ("to", int)], edge_values)
                self.edge_data[edge_type] = Table(Index(schema))

            self.edges[edge_type] = SparseTensor([0, 0])

        self.max_id = 0

    def add_node(self, node_type, key, value=tuple()):
        self.nodes[node_type].insert(key + value)

        self.node_ids.insert([node_type, key, self.max_id])
        self.max_id += 1

        for edges in self.edges.values():
            edges.expand([self.max_id, self.max_id])

    def add_edge(self, label, from_node, to_node, value=[]):
        (from_type, from_key) = from_node
        [index_from] = list(self.node_ids.slice({
            "type": from_type, "key": from_key}).select(["id"]))

        (to_type, to_key) = to_node
        [index_to] = list(self.node_ids.slice({
            "type": to_type, "key": to_key}).select(["id"]))

        self.edges[label][(index_from, index_to)] = 1
        if self.edge_data[label] is not None:
            self.edge_data[label].insert([index_from, index_to] + value)

    def bft(self, edge_type, node):  # breadth-first traversal
        edges = self.edges[edge_type]

        (node_type, node_key) = node
        
        [node_index] = list(self.node_ids.slice({
            "type": node_type, "key": node_key}).select(["id"]))

        adjacent = SparseTensor([self.max_id], np.bool)
        adjacent[node_index] = True

        visited = SparseTensor([self.max_id], np.bool)

        while adjacent.any():
            visited = visited | adjacent
            adjacent = einsum('ji,j->i', edges, adjacent).copy()
            adjacent.mask(visited)

            for (i,), _ in adjacent.filled():
                for node_type, node_key, _ in self.node_ids.slice({"id": i}):
                    yield node_type, node_key


def test_bft():
    g = Graph({"node": ([("key", int)], [])}, {"edge": None})

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


if __name__ == "__main__":
    test_bft()
    print("PASS")

