import pydecode.hyper as ph
import pydecode.chart as chart
import interface

class Chart:
    def __init__(self, score = None):
        self.chart = \
            chart.ChartBuilder(lambda a:a, chart.HypergraphSemiRing,
                               build_hypergraph=True)
        self.score = score

    def initialize(self, item, score=0.0):
        self.chart.init(item)
        # self.chart[item] = ChartItem(score, None)

    # def has(self, item):
    #     return item in self.chart


    def set(self, item, vals):
        seq = (reduce(lambda x, y: x * y, [self.chart[v] for v in vs] +
                ([self.chart.sr(score)] if score != 0.0 else []))
               for vs, score in vals)
        self.chart[item] = self.chart.sum(seq)

            # self.chart[item] = v
        # e;se
        # self.chart[item] = None

    def regen(self):
        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_vector((self.score(edge.label) if edge.label else 0.0
                          for edge in self.hypergraph.edges ))

        path = ph.best_path(self.hypergraph, self.pot)
        return [node.label for node in path.nodes]

    def backtrace(self, item):
        self.hypergraph = self.chart.finish()
        self.pot = ph.LogViterbiPotentials(self.hypergraph) \
            .from_vector((self.score(edge.label) if edge.label else 0.0
                          for edge in self.hypergraph.edges ))

        path = ph.best_path(self.hypergraph, self.pot)
        return [node.label for node in path.nodes]
        # items = []
        # def collect(item, depth):
        #     items.append(item)
        #     if self.chart[item].bp:
        #         for item in self.chart[item].bp:
        #             collect(item, depth+1)
        # collect(item, 0)
        # return items

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
    c = Chart()
    interface.Parser().parse_bigram(sent_len, scorer, None, c)
    hypergraph = c.hypergraph
    pot = c.pot


    def score(arc):
        if isinstance(arc, interface.Arc):
            return scorer.arc_score(arc.head, arc.mod)
        elif isinstance(arc, interface.Bigram):
            return scorer.bigram_score(arc.s1, arc.s2) - \
                ((arc.s2 - arc.s1 - 1) * scorer.skip_penalty) if arc.s1 != arc.s2 else 0.0
        else:
            return 0.0

    def f(pen):
        scorer.skip_penalty = pen
        c.score = score
        parse = interface.make_parse(sent_len+1, c.regen())
        return sent_len - parse.skipped_words()

    pen = binary_search(f, m)
    scorer.skip_penalty = pen
    return interface.make_parse(sent_len+1, c.regen())
