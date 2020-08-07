from btree.table import Index, Schema, Table
from einsum.einsum import einsum
from tensor.sparse import SparseTensor


class Graph(object):
    def __init__(self):
        self.nodes = Table(Index(Schema([("id", int)], [("node", int)])))
        self.nodes.add_index("node_value", ("node",))
        self.edges = SparseTensor([0, 0])

    def add_node(self, node):
        self.nodes.insert([len(self.nodes), node])
        self.edges.expand([len(self.nodes), len(self.nodes)])

    def add_edge(self, node_from, node_to):
        [index_from] = list(self.nodes.slice({"node": node_from}).select(["id"]))
        [index_to] = list(self.nodes.slice({"node": node_to}).select(["id"]))
        self.edges[(index_from, index_to)] = 1

    def bft(self, node):  # breadth-first traversal
        [node_index] = list(self.nodes.slice({"node": node}).select(["id"]))
        adjacent = SparseTensor([len(self.nodes)])
        visited = SparseTensor([len(self.nodes)])

        while adjacent.any():
            visited = visited | adjacent
            adjacent = einsum('ji,j->i', self.edges, adjacent)
            adjacent.mask(visited)
            print("bft", adjacent)


def test_bft():
    g = Graph()
    g.add_node(10)
    g.add_node(20)
    g.add_node(30)
    g.add_node(40)
    g.add_node(50)
    g.add_node(60)
    g.add_edge(10, 20)
    g.add_edge(20, 30)
    g.add_edge(30, 40)
    g.add_edge(30, 50)
    g.bft(10)


if __name__ == "__main__":
    test_bft()
    print("PASS")
