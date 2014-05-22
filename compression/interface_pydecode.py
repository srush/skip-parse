import pydecode.hyper as ph
import pydecode.chart as chart
import interface
import numpy as np

class Chart:
    def __init__(self, n, m=None, score=None):
        self.chart = \
            ph.DPChartBuilder(build_hypergraph=True, strict=True)
        hasher = ph.SizedTupleHasher([8, 2, n+1, n+1, n+1])
        self.chart.set_hasher(hasher)
            # chart.ChartBuilder(lambda a:a, chart.HypergraphSemiRing,
            #                    build_hypergraph=True)
        self.score = score
        self.m = m
        self.n = n -1
    def initialize(self, item, score=0.0):
        self.chart.init(item)
        # self.chart[item] = ChartItem(score, None)

    # def has(self, item):
    #     return item in self.chart


    def set(self, item, vals):
        self.chart.set(item, vals)
        # seq = (reduce(lambda x, y: x * y, [self.chart[v] for v in vs] +
        #         ([self.chart.sr(score)] if score != 0.0 else []))
        #        for vs, score in vals)
        # self.chart[item] = self.chart.sum(seq)

            # self.chart[item] = v
        # e;se
        # self.chart[item] = None

    def regen(self, penalty, counts):
        # scores = np.zeros(len(self.hypergraph.edges))
        # for edge_num, label, tail_labels in self.hypergraph.node_labels():
        #     #scores[edge_num] = self.score(label, len(self.hypergraph.edges[edge_num].tail))
        #     #len(self.hypergraph.edges[edge_num].tail)
        #     #scores[edge_num] = self.score(label, len(tail_labels))
        #     typ, d, s, t, _ = label
        #     if typ == interface.Trap:
        #         if d == interface.Left: s, t = t, s
        #         scores[edge_num] = scorer.arc_score(s, t)
        #     if typ == interface.Tri and d == interface.Right and len(tail_labels) == 1:
        #         scores[edge_num] =  scorer.bigram_score(s, t+1) - \
        #             ((t+1 - s - 1) * scorer.skip_penalty) if s != t+1 else 0.0

        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_array(self.scores + (penalty * counts))
        path = ph.best_path(self.hypergraph, self.pot)
        return [node.label for node in path.nodes]

    def counts(self):
        counts = np.zeros(len(self.hypergraph.edges), dtype=np.int32)
        for edge_num, label in self.hypergraph.head_labels():
            counts[edge_num] = 1 if (label[0] == interface.Trap) else 0
        return counts

    def backtrace(self, item):
        self.hypergraph = self.chart.finish()

        scores = np.zeros(len(self.hypergraph.edges))
        self.skips = np.zeros(len(self.hypergraph.edges))
        self.scores = scores

        for edge_num, label, tail_labels in self.hypergraph.node_labels():
            scores[edge_num] = self.score(label, len(tail_labels))
        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_array(scores)

        if self.m != None:
            counts = np.zeros(len(self.hypergraph.edges), dtype=np.int32)
            if self.m > (self.n / 2):
                m2 = self.n - self.m
                for edge_num, label, tail_labels in self.hypergraph.node_labels():
                    typ, d, s, t, _ = label
                    if typ == interface.Tri and d == interface.Right and len(tail_labels) == 1:
                        if t != s:
                            counts[edge_num] = (t - s)

                self.counts = ph.CountingPotentials(self.hypergraph) \
                    .from_array(counts)
                hmap = ph.extend_hypergraph_by_count(self.hypergraph, self.counts, 0, m2, m2)
            else:
                for edge_num, label in self.hypergraph.head_labels():
                    counts[edge_num] = 1 if (label[0] == interface.Trap) else 0

                self.counts = ph.CountingPotentials(self.hypergraph) \
                    .from_array(counts)
                hmap = ph.extend_hypergraph_by_count(self.hypergraph, self.counts, 0, self.m, self.m)

            new_pot = self.pot.up_project(hmap.domain_hypergraph, hmap)
            path = ph.best_path(hmap.domain_hypergraph, new_pot)
        else:
            path = ph.best_path(self.hypergraph, self.pot)
        return [node.label for node in path.nodes]


def parse_bigram(sent_len, scorer, m=None):
    c = Chart(sent_len + 1, m)
    return interface.Parser().parse_bigram(sent_len, scorer, None, c)
    # hypergraph = c.hypergraph
    # return interface.make_parse(sent_len+1, c.regen())

def parse_binary_search(sent_len, scorer, m):
    def binary_search(seq, t):
        min = -10
        max = 10
        for i in range(25):
            if max < min:
                return -1
            m = (min + max) / 2.0

            size = seq(m)
            print t, m, size
            if size < t:
                min = m
            elif size > t:
                max = m
            else:
                return m
        return m
    c = Chart(sent_len+1)
    interface.Parser().parse_bigram(sent_len, scorer, None, c)
    counts = c.counts()

    # hypergraph = c.hypergraph
    # pot = c.pot


    # def score(label, tail_size):
    #     typ, d, s, t, _ = label
    #     if typ == interface.Trap:
    #         if d == interface.Left: s, t = t, s
    #         return scorer.arc_score(s, t)
    #     if typ == interface.Tri and d == interface.Right and tail_size == 1:
    #         return scorer.bigram_score(s, t+1) - \
    #             ((t+1 - s - 1) * scorer.skip_penalty) if s != t+1 else 0.0
    #     return 0.0

    def f(pen):
        scorer.skip_penalty = pen
        # c.score = score
        parse = interface.make_parse(sent_len+1, c.regen(pen, counts))
        return sent_len - parse.skipped_words()

    pen = binary_search(f, m)
    scorer.skip_penalty = pen
    return interface.make_parse(sent_len+1, c.regen(pen, counts))
