
class ChartItem(namedtuple("ChartItem", ["score", "bp"])):
    """
    An item in the chart.

    Attributes
    -----------
    score : float
       The score of the current cell.

    bp : item
       A back-pointer to the best prev item.
    """
    pass

class Chart:
    """
    A general dynamic programming chart.
    """
    def __init__(self, score=lambda a: a):
        self.chart = {}
        self.score = score

    def Node(self, *args):
        return args

    def initialize(self, item, score=0.0):
        """
        Initialize an item in the chart.

        Parameters
        ----------
        item :

        score : float
            The initial score of the item
        """
        self.chart[item] = ChartItem(score, None)

    # def has(self, item):
    #     return item in self.chart

    def set(self, item, vals):
        """
        Add an item to the chart.

        Parameters
        ----------
        item :

        vals : list of tuples
            The list of pairs.
            First element in the pair is a list of items (bp's).
            Second element in the pair is a score (edge score).

        """

        seq = (((sum((self.chart[v].score for v in vs)) + self.score(score)), vs)
               for vs, score in vals )

        v = ChartItem(*max(itertools.chain([(-1e9, None)], seq),
                           key = lambda a: a[0]))
        if v[1] is not None:
            self.chart[item] = v
        # e;se
        # self.chart[item] = None

    def backtrace(self, item):
        """
        Construct a back trace from a chart.

        Parameters
        ----------
        item :
            The item to start from.

        Returns
        ---------
        items : list of items
            The items in the best structure.
        """
        items = []
        def collect(item, depth):
            assert depth < 50, item
            items.append(item)
            if self.chart[item].bp:
                for item in self.chart[item].bp:
                    collect(item, depth+1)
        collect(item, 0)
        return items

class Arc(namedtuple("Arc", ["head", "mod", "sibling"])):
    def __new__(cls, head, mod, sibling=None):
        return super(Arc, cls).__new__(cls, head, mod, sibling)

class Bigram(namedtuple("Bigram", ["s1", "s2"])):
    pass

def Edge(a, b=0.0):
    return (a, b)

class OldInterface(object):
        def parse_bigram(self, sent_len, scorer, m=None, chart=None):
        """
        Parses with bigrams.

        Parameters
        -----------
        n : int
           The length of the sentence.

        scorer : Scorer
           The arc-factored weights and bigram scores.

        m : int
           The length of the compressed sentence.
           None if any length allowed.

        Returns
        -------
        parse : Parse
           The best dependency parse with these constraints.
        """
        n = sent_len + 1

        def score(arc):
            if isinstance(arc, Arc):
                return scorer.arc_score(arc.head, arc.mod)
            elif isinstance(arc, Bigram):
                return scorer.bigram_score(arc.s1, arc.s2) - \
                    ((arc.s2 - arc.s1 - 1) * scorer.skip_penalty) if arc.s1 != arc.s2 else 0.0
            else:
                return 0.0

        def score2(label, tail_size):
            typ, d, s, t, _ = label
            if typ == Trap:
                if d == Left: s, t = t, s
                return scorer.arc_score(s, t)
            if typ == Tri and d == Right and tail_size == 1:
                return scorer.bigram_score(s, t+1) - \
                    ((t+1 - s - 1) * scorer.skip_penalty) if s != t+1 else 0.0


            return 0.0
            # if typ == Tri and d == Right:
            #     return scorer.bigram_score(arc.s1, arc.s2) - \
            #         ((arc.s2 - arc.s1 - 1) * scorer.skip_penalty) if arc.s1 != arc.s2 else 0.0

            # if isinstance(arc, Arc):
            #     return scorer.arc_score(arc.head, arc.mod)
            # elif isinstance(arc, Bigram):
            #     return scorer.bigram_score(arc.s1, arc.s2) - \
            #         ((arc.s2 - arc.s1 - 1) * scorer.skip_penalty) if arc.s1 != arc.s2 else 0.0
            # else:
            #     return 0.0

        if chart != None:
            c = chart
            c.score = score2
        else:
            c = Chart(score)
            c.score = score

        diff = n
        any_size = (m is None)
        if m is not None:
            diff = n - m


        # Initialize the chart.
        [c.initialize(NodeType(sh, d, s, s, 0), 0.0)
         for s in range(n)
         for d in [Right, Left]
         for sh in ([Tri] if d == Left else [TriSkipped])]

        for s in range(n):
            c.set(NodeType(Tri, Right, s, s, 0),
                  [Edge([(TriSkipped, Right, s, s, 0)], Bigram(s, s+1))])

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                remaining = n - s

                if any_size:
                    mod_counts = [0]
                else:
                    mod_counts = range(m - remaining - 1, m + 1)

                for mod_count in mod_counts:

                    # First create incomplete items.
                    if s != 0 and (mod_count > 0 or any_size):
                        edges = [Edge((key1, key2), Arc(t, s))
                               for r  in range(s, t)
                               for m1 in (range(mod_count) if not any_size else [0])
                               for key1 in [(Tri, Right, s, r, m1)]
                               if key1 in c.chart
                               for m2 in ([mod_count - m1 - 1] if not any_size else [0])
                               for key2 in [(Tri, Left, r+1, t, m2)]
                               if key2 in c.chart]
                        c.set(NodeType(Trap, Left, s, t, mod_count),
                              edges)


                    if mod_count > 0 or any_size:
                        c.set(NodeType(Trap, Right, s, t, mod_count),
                              [Edge([key1, key2], Arc(s, t))
                               for r  in range(s, t)
                               for m1 in (range(mod_count) if not any_size else [0])
                               for key1 in [(Tri, Right, s, r, m1)]
                               if key1 in c.chart
                               for m2 in ([mod_count - m1 - 1] if not any_size else [0])
                               for key2 in [(Tri, Left, r+1, t, m2)]
                               if key2 in c.chart])


                    if s != 0:
                        c.set(NodeType(Tri, Left, s, t, mod_count),
                              [Edge([key1, key2])
                               for r  in range(s, t)
                               for m1 in (range(mod_count + 1) if not any_size else [0])
                               for key1 in [(Tri, Left, s, r, m1)]
                               if key1 in c.chart
                               for m2 in ([mod_count - m1] if not any_size else [0])
                               for key2 in [(Trap, Left, r, t, m2)]
                               if key2 in c.chart]
                              )

                    c.set(NodeType(Tri, Right, s, t, mod_count),
                          [Edge([key1, key2])
                           for r in range(s + 1, t + 1)
                           for m1 in (range(mod_count + 1) if not any_size else [0])
                           for key1 in [(Trap, Right, s, r, m1)]
                           if key1 in c.chart
                           for m2 in ([mod_count - m1] if not any_size else [0])
                           for key2 in [(Tri, Right, r, t, m2)]
                           if key2 in c.chart] +
                          ([Edge([key], Bigram(s, t+1))
                           for key in [(TriSkipped, Right, s, s, 0)]]
                           if mod_count  == 0 else []))

        return make_parse(n, c.backtrace(NodeType(Tri, Right, 0, n-1, m if not any_size else 0)))


    def parse_second_bigram(self, sent_len, scorer, chart=None):
        """
        Parses with bigrams.

        Parameters
        -----------
        n : int
           The length of the sentence.

        scorer : Scorer
           The arc-factored weights and bigram scores.

        Returns
        -------
        parse : Parse
           The best dependency parse with these constraints.
        """
        n = sent_len + 1

        def score(arc):
            if isinstance(arc, Arc):
                return scorer.arc_score(arc.head, arc.mod, arc.sibling)
            elif isinstance(arc, Bigram):
                if arc.s1 != arc.s2:
                    return scorer.bigram_score(arc.s1, arc.s2) - \
                        ((arc.s2 - arc.s1 - 1) * scorer.skip_penalty)
                else:
                    return 0.0
            else:
                return 0.0

        if chart != None:
            c = chart
        else:
            c = Chart(score)
        c.score = score

        # Initialize the chart.
        [c.initialize(NodeType(sh, d, s, s, 0), 0.0)
         for s in range(n)
         for d in [Right, Left]
         for sh in ([Tri] if d == Left else [TriSkipped1])]

        for s in range(n):
            c.set(NodeType(TriSkipped, Right, s, s, 0),
                  [Edge([(TriSkipped1, Right, s, s, 0)], Bigram(s, s+1))])

            c.set(NodeType(Tri, Right, s, s, 0),
                  [Edge([(TriSkipped, Right, s, s, 0)])])

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)

                if s != 0:
                    c.set(NodeType(Box, Left, s, t, 0),
                          (Edge([key1, key2])
                           for r  in range(s, t)
                           for key1 in [(Tri, Right, s, r, 0)]
                           if key1 in c.chart
                           for key2 in [(Tri, Left, r+1, t, 0)]
                           if key2 in c.chart))

                if s != 0:
                    c.set(NodeType(Trap, Left, s, t, 0),
                          [Edge([key1, key2], Arc(t, s, t))
                           for key1 in [(Tri, Right, s, t-1, 0)]
                           if key1 in c.chart
                           for key2 in [(Tri, Left, t, t, 0)]
                           if key2 in c.chart] +
                          [Edge([key1, key2], Arc(t, s, r))
                           for r in range(s+1, t)
                           for key1 in [(Box, Left, s, r, 0)]
                           if key1 in c.chart
                           for key2 in [(Trap, Left, r, t, 0)]
                           if key2 in c.chart])

                c.set(NodeType(Trap, Right, s, t, 0),
                      [Edge([key1, key2], Arc(s, t, s))
                       for r in range(s, t)
                       for key1 in [(TriSkipped, Right, s, r, 0)]
                       if key1 in c.chart
                       for key2 in [(Tri, Left, r+1, t, 0)]
                       if key2 in c.chart] +
                      [Edge([key1, key2], Arc(s, t, r))
                       for r  in range(s + 1, t)
                       for key1 in [(Trap, Right, s, r, 0)]
                       if key1 in c.chart
                       for key2 in [(Box, Left, r, t, 0)]
                       if key2 in c.chart])

                if s != 0:
                    c.set(NodeType(Tri, Left, s, t, 0),
                          (Edge([key1, key2])
                           for r  in range(s, t)
                           for key1 in [(Tri, Left, s, r, 0)]
                           if key1 in c.chart
                           for key2 in [(Trap, Left, r, t, 0)]
                           if key2 in c.chart))

                c.set(NodeType(TriSkipped, Right, s, t, 0),
                      [Edge([key], Bigram(s, t+1))
                       for key in [(TriSkipped1, Right, s, s, 0)]])


                c.set(NodeType(Tri, Right, s, t, 0),
                      ([Edge([key1, key2])
                        for r in range(s + 1, t + 1)
                        for key1 in [(Trap, Right, s, r, 0)]
                        if key1 in c.chart
                        for key2 in [(Tri, Right, r, t, 0)]
                        if key2 in c.chart] +
                      [Edge([key])
                       for key in [(TriSkipped, Right, s, t, 0)]]))

        c.set("final",
              [Edge([NodeType(t, Right, 0, n-1, 0)])
               for t in [Tri, TriSkipped]])
        print c.chart["final"]
        return make_parse(n, c.backtrace("final"))


    def parse_skip(self, sentence_length, scorer, m, chart=None):
        """
        Parses with skips.

        Parameters
        -----------
        sentence_length : int
           The length of the sentence.

        scorer : Scorer
           The arc-factored weights on each possible dependency.

        m : int
           The length of the compressed sentence.

        Returns
        -------
        parse : Parse
           The best dependency parse with these constraints.
        """
        if chart != None:
            c = chart
        else:
            c = Chart(lambda arc: scorer.arc_score(arc.head, arc.mod) if arc != 0.0 else 0.0)

        n = sentence_length + 1

        # Add terminal nodes.
        [c.initialize(NodeType(sh, d, s, s, 0), 0.0)
         for s in range(n)
         for d in [Right, Left]
         for sh in [Trap, Tri]]

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)
                remaining = n - s
                need = m

                for mod_count in range(m - remaining -1, m + 1):

                    # First create incomplete items.
                    if s != 0 and mod_count > 0:
                        c.set(NodeType(Trap, Left, s, t, mod_count),
                              [Edge([key1, key2], Arc(t, s))
                               for r in range(s, t)
                               for m1 in range(mod_count)
                               for key1 in [(Tri, Right, s, r, m1)]
                               if key1 in c.chart
                               for m2 in [mod_count  - m1 - 1]
                               for key2 in [(Tri, Left, r+1, t, m2)]
                               if key2 in c.chart
                               ])

                    if mod_count > 0:
                        c.set(NodeType(Trap, Right, s, t, mod_count),
                              [Edge([key1, key2], Arc(s, t))
                               for r in range(s, t)
                               for m1 in range(mod_count)
                               for key1 in [(Tri, Right, s, r, m1)]
                               if key1 in c.chart
                               for m2 in [mod_count - m1 - 1]
                               for key2 in [(Tri, Left, r+1, t, m2)]
                               if key2 in c.chart
                               ])


                    c.set(NodeType(TrapSkipped, Right, s, t, mod_count),
                          [Edge([key1, key2])
                           for m1 in range(mod_count + 1)
                           for key1 in [(Tri, Right, s, t-1, m1)]
                           if key1 in c.chart
                           for m2 in [mod_count - m1]
                           for key2 in [(Tri, Left, t, t, m2)]
                           if key2 in c.chart
                           ])

                    if s != 0:
                        c.set(NodeType(Tri, Left, s, t, mod_count),
                              [Edge([key1, key2])
                               for r in range(s, t)
                               for m1 in range(mod_count + 1 )
                               for key1 in [(Tri, Left, s, r, m1)]
                               if key1 in c.chart
                               for m2 in [mod_count - m1]
                               for key2 in [(Trap, Left, r, t, m2)]
                               if key2 in c.chart])

                    c.set(NodeType(Tri, Right, s, t, mod_count),
                          [Edge([key1, key2])
                           for r in range(s + 1, t + 1)
                           for m1 in range(mod_count + 1)
                           for key1 in [(Trap, Right, s, r, m1)]
                           if key1 in c.chart
                           for m2 in [mod_count - m1]
                           for key2 in [(Tri, Right, r, t, m2)]
                           if key2 in c.chart] + \
                          [Edge([key1, key2])
                           for m1 in range(mod_count + 1)
                           for key1 in [(TrapSkipped, Right, s, t, m1)]
                           if key1 in c.chart
                           for m2 in [mod_count  - m1]
                           for key2 in [(Tri, Right, t, t, m2)]
                           if key2 in c.chart
                           ])

        return make_parse(n, c.backtrace(NodeType(Tri, Right, 0, n-1, m)))

    def parse_binary_search(self, sent_len, scorer, m):
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
        def f(pen):
            scorer.skip_penalty = pen
            parse = self.parse_bigram(sent_len, scorer, None)
            return sent_len - parse.skipped_words()

        pen = binary_search(f, m)
        scorer.skip_penalty = pen
        return self.parse_bigram(sent_len, scorer, None)
