import pydecode.hyper as ph
import pydecode.chart as chart
import interface
import numpy as np
import time

class Chart:
    def __init__(self, n, m=None, score=None):
        self.chart = \
            ph.DPChartBuilder(build_hypergraph=True, strict=False)
        hasher = ph.QuartetHash(ph.Quartet(interface.kShapes, interface.kDirs, n+1, n+1))
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
        #print self.scores
        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_array(self.scores + (penalty * counts))

    def unconstrained_search(self):
        path = ph.best_path(self.hypergraph, self.pot, chart=self._internal_chart)
        return [node.label.unpack() for node in path.nodes]

    def constrained_search(self):
        self.skips = np.zeros(len(self.hypergraph.edges))
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
            return [node.label.unpack() for node in path.nodes]
        else:
            return []


    def backtrace(self, item):
        self.hypergraph = self.chart.finish(False)
        self._internal_chart = ph.LogViterbiChart(self.hypergraph)
        return self.constrained_search()


def parse_bigram(sent_len, scorer, m):
    c = Chart(sent_len + 1, m)
    return interface.Parser().parse_bigram_any(sent_len, scorer, c)

def parse_binary_search(sent_len, scorer, m, limit=10, min_val=-10, max_val=10):
    def binary_search(seq, t):
        min = min_val
        max = max_val
        for i in range(limit):
            if max < min:
                return -1, False
            m = (min + max) / 2.0

            size = seq(m)
            if size < t:
                min = m
            elif size > t:
                max = m
            else:
                return m, True
        return m, False

    c = Chart(sent_len+1)


    interface.Parser().parse_bigram_any(sent_len, scorer, c)


    def f(pen):
        scorer.skip_penalty = pen
        c.regen(pen, c.counts)
        parse = interface.make_parse(sent_len+1, c.unconstrained_search())
        return sent_len - parse.skipped_words()

    pen, success = binary_search(f, m)
    if success:
        scorer.skip_penalty = pen
        c.regen(pen, c.counts)
        return interface.make_parse(sent_len+1, c.unconstrained_search())
    else:
        scorer.skip_penalty = 0.0
        c.regen(0.0, c.counts)
        return interface.make_parse(sent_len+1, c.constrained_search())
