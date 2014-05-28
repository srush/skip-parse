
import pydecode.hyper as ph
import skipdep.interface as interface
import numpy as np

class Chart:
    def __init__(self, n):
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
        self.n = n - 1

    def initialize(self, item, score=0.0):
        self.chart.init(item)

    def set(self, item, vals):
        self.chart.set(item, vals)

    def reweight(self, penalty):
        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_array(self.scores + (penalty * self.counts))

    def unconstrained_search(self):
        path = ph.best_path(self.hypergraph, self.pot, chart=self._internal_chart)
        return [node.label.unpack() for node in path.nodes]

    def constrained_search(self, m):
        if m != None:
            if m < (self.n / 2):
                counts = ph.CountingPotentials(self.hypergraph) \
                    .from_array(self.counts)
                path = ph.count_constrained_viterbi(self.hypergraph, self.pot, counts, m)
            else:
                counts = ph.CountingPotentials(self.hypergraph) \
                    .from_array(self.reverse_counts)

                path = ph.count_constrained_viterbi(self.hypergraph, self.pot, counts, self.n - m)
            return [node.label.unpack() for node in path.nodes]
        else:
            return self.unconstrained_search()

    def finish(self):
        self.hypergraph = self.chart.finish(False)
        self._internal_chart = ph.LogViterbiChart(self.hypergraph)
        self.reweight(0.0)

class Bisector(object):
    def __init__(self, min_val=-10, max_val=10, limit=10):
        self.min_val = min_val
        self.max_val = max_val
        self.limit = limit

    def run(self, f, target):
        cur_min = self.min_val
        cur_max = self.max_val
        self.history = []
        for i in range(self.limit):
            if cur_max < cur_min:
                return -1, False
            m = (cur_min + cur_max) / 2.0
            result = f(m)
            self.history.append((m, result, target))
            if result < target:
                cur_min = m
            elif result > target:
                cur_max = m
            else:
                return True

        return False


def parse_bigram(sent_len, scorer, m):
    n = sent_len + 1
    c = Chart(n)
    interface.Parser().parse_bigram(sent_len, scorer, c)
    c.finish()
    return interface.make_parse(n, c.constrained_search(m))

def parse_second_bigram(sent_len, scorer, m):
    n = sent_len + 1
    c = Chart(n)
    interface.Parser().parse_second_bigram(sent_len, scorer, c)
    c.finish()
    return interface.make_parse(n, c.constrained_search(m))

def parse_binary_search(sent_len, scorer, m,
                        searcher, order=1):
    n = sent_len + 1
    c = Chart(n)
    if order == 1:
        interface.Parser().parse_bigram(sent_len, scorer, c)
    elif order == 2:
        interface.Parser().parse_second_bigram(sent_len, scorer, c)
    c.finish()

    def f(pen):
        c.reweight(pen)
        parse = interface.make_parse(n, c.unconstrained_search())
        return sent_len - parse.skipped_words()
    success = searcher.run(f, m)

    if success:
        return interface.make_parse(n,
                                    c.unconstrained_search())
    else:
        c.reweight(0.0)
        return interface.make_parse(n,
                                    c.constrained_search(m))
