import pydecode.hyper as ph
import pydecode.chart as chart
import interface
import numpy as np
import time

class Chart:
    def __init__(self, n, m=None, score=None):
        self.chart = \
            ph.DPChartBuilder(build_hypergraph=True, strict=False)
        # First
        # hasher = ph.SizedTupleHasher([kShapes, kDirs, n+1, n+1, n+1])
        # self.chart.set_hasher(hasher)

        # Second
        # quart = ph.Quartet(interface.kShapes, interface.kDirs, n+1, n+1)
        hasher = ph.QuartetHash(ph.Quartet(interface.kShapes, interface.kDirs, n+1, n+1))
        # hasher = ph.SizedTupleHasher([interface.kShapes, interface.kDirs, n+1, n+1])
        # print hasher.max_size()
        num_edges = 5 * n ** 3
        self.chart.set_hasher(hasher)
        self.scores = np.zeros([num_edges])
        self.counts = np.zeros([num_edges], dtype=np.int32)
        self.reverse_counts = np.zeros([num_edges], dtype=np.int32)
        self.chart.set_data([self.scores, self.counts, self.reverse_counts])
        self.chart.set_expected_size(hasher.max_size(), num_edges, max_arity=2)


        self.Node = ph.Quartet


        self.score = score
        self.m = m
        self.n = n - 1

    def initialize(self, item, score=0.0):
        self.chart.init(item)

    def set(self, item, vals):
        self.chart.set(item, vals)

    def regen(self, penalty, counts):
        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_array(self.scores + (penalty * counts))
        path = ph.best_path(self.hypergraph, self.pot, chart = self._internal_chart)
        return [node.label for node in path.nodes]

    def get_counts(self):
        return self.counts
        # counts = np.zeros(len(self.hypergraph.edges), dtype=np.int32)
        # for edge_num, label in self.hypergraph.head_labels():
        #     counts[edge_num] = 1 if (label[0] == interface.Trap) else 0
        # return counts

    def backtrace(self, item):

        self.hypergraph = self.chart.finish(False)
        self._internal_chart = ph.LogViterbiChart(self.hypergraph)

        self.skips = np.zeros(len(self.hypergraph.edges))
        # self.scores = scores

        # for edge_num, label, tail_labels in self.hypergraph.node_labels():
        #     scores[edge_num] = self.score(label, len(tail_labels))
        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_array(self.scores)


        if self.m != None:
            if self.m < (self.n / 2):
                counts = ph.CountingPotentials(self.hypergraph) \
                    .from_array(self.counts)

                path = ph.count_constrained_viterbi(self.hypergraph, self.pot, counts, self.m)
            else:
                counts = ph.CountingPotentials(self.hypergraph) \
                    .from_array(self.reverse_counts)

                path = ph.count_constrained_viterbi(self.hypergraph, self.pot, counts, self.n - self.m)

            # counts = np.zeros(len(self.hypergraph.edges), dtype=np.int32)
            #
            #     m2 = self.n - self.m
            #     for edge_num, label, tail_labels in self.hypergraph.node_labels():
            #         typ, d, s, t, _ = label
            #         if typ == interface.Tri and d == interface.Right and len(tail_labels) == 1:
            #             if t != s:
            #                 counts[edge_num] = (t - s)

            #     self.counts = ph.CountingPotentials(self.hypergraph) \
            #         .from_array(counts)
            #     hmap = ph.extend_hypergraph_by_count(self.hypergraph, self.counts, 0, m2, m2)
            # else:
            #     for edge_num, label in self.hypergraph.head_labels():
            #         counts[edge_num] = 1 if (label[0] == interface.Trap) else 0

            #     self.counts = ph.CountingPotentials(self.hypergraph) \
            #         .from_array(counts)
            # hmap = ph.extend_hypergraph_by_count(self.hypergraph, counts, 0, self.m, self.m)

            # new_pot = self.pot.up_project(hmap.domain_hypergraph, hmap)
            # path = ph.best_path(hmap.domain_hypergraph, new_pot)




            # for edge in path.edges:
            #     print edge.id, edge.head
            # for n in path.nodes:
            #     print n.id

            # for n in path.nodes:
            #     print n.label.unpack()

            # new_pot = self.pot.up_project(hmap.domain_hypergraph, hmap)
            # path = ph.best_path(hmap.domain_hypergraph, new_pot)

        else:
            path = ph.best_path(self.hypergraph, self.pot)
        return [node.label.unpack() for node in path.nodes]


def parse_bigram(sent_len, scorer, m):
    c = Chart(sent_len + 1, m)
    return interface.Parser().parse_bigram_any(sent_len, scorer, c)

def parse_binary_search(sent_len, scorer, m):
    def binary_search(seq, t):
        min = -10
        max = 10
        for i in range(25):
            if max < min:
                return -1
            m = (min + max) / 2.0

            size = seq(m)
            if size < t:
                min = m
            elif size > t:
                max = m
            else:
                return m
        return m
    c = Chart(sent_len+1)
    print sent_len+1
    tim = time.time()
    interface.Parser().parse_bigram_any(sent_len, scorer, c)
    print "construct", time.time() - tim
    counts = c.counts

    def f(pen):
        scorer.skip_penalty = pen
        parse = interface.make_parse(sent_len+1, c.regen(pen, counts))
        return sent_len - parse.skipped_words()


    tim = time.time()
    pen = binary_search(f, m)
    scorer.skip_penalty = pen
    print "Search", time.time() - tim
    return interface.make_parse(sent_len+1, c.regen(pen, counts))
