from skipdep.parse import Parse, Scorer

# Globals
kShapes = 6
Tri = 0
TrapSkipped = 1
Trap = 2
TriSkipped = 3
Box = 4
TriSkipped1 = 5

kDirs = 2
Right = 0
Left = 1

def NodeType(type, dir, span, count, states=None) :
    if states is None:
        return (type, dir, span, count)
    return (type, dir, span, count, states)
def node_type(nodetype): return nodetype[0]
def node_span(nodetype): return nodetype[2:4]
def node_dir(nodetype): return nodetype[1]

def make_parse(n, back_trace):
    """
    Construct the best parse from a back trace.

    Parameters
    -------------
    n : Length of the parse.

    back_trace : list of NodeType's
       The backtrace from the parse chart.

    Returns
    ---------
    parse : Parse
       A parse object.
    """
    mods = [None] * n
    mods[0] = -1
    for element in back_trace:
        if isinstance(element, tuple) and  node_type(element) == Tri:
            span = node_span(element)

        if isinstance(element, tuple) and  node_type(element) == Trap:
            span = node_span(element)
            if node_dir(element) == Right:
                mods[span[1]] = span[0]
            if node_dir(element) == Left:
                mods[span[0]] = span[1]
    return Parse(mods)

class Parser(object):
    def _initialize(self, c, n, scorer):
        # Initialize the chart.
        [c.initialize(c.Node(sh, d, s, s), 0.0)
         for s in range(n)
         for d in [Right, Left]
         for sh in ([Tri] if d == Left else [TriSkipped1])]

        for s in range(n):
            c.set(c.Node(TriSkipped, Right, s, s),
                  [([c.Node(TriSkipped1, Right, s, s)],
                    [scorer.bigram_score(s, s+1, True), 0, 0])])

            c.set(c.Node(Tri, Right, s, s),
                  [([c.Node(TriSkipped, Right, s, s)], [0,0,0])])


    def parse_bigram(self, sent_len, scorer, chart=None,
                     single_root=True):
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
        c = chart

        self._initialize(c, n, scorer)

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break

                # First create incomplete items.
                if s != 0:
                    arc_score = scorer.arc_score(t, s)
                    c.set(c.Node(Trap, Left, s, t),
                          [((c.Node(Tri, Right, s, r),
                             c.Node(Tri, Left, r+1, t)), [arc_score, 1, 0])
                           for r in xrange(s, t)])

                arc_score = scorer.arc_score(s, t)
                c.set(c.Node(Trap, Right, s, t),
                      [((c.Node((Tri if s != 0 else TriSkipped),
                                Right, s, r),
                         c.Node(Tri, Left, r+1, t)), [arc_score, 1, 0])
                       for r in xrange(s, t)])

                if s != 0:
                    c.set(c.Node(Tri, Left, s, t),
                          [((c.Node(Tri, Left, s, r),
                             c.Node(Trap, Left, r, t)), [0.0, 0, 0])
                           for r in xrange(s, t)])

                c.set(c.Node(TriSkipped, Right, s, t),
                      [([c.Node(TriSkipped1, Right, s, s)],
                        [scorer.bigram_score(s, t+1), 0, t - s])])

                c.set(c.Node(Tri, Right, s, t),
                      [((c.Node(Trap, Right, s, r),
                         c.Node(Tri, Right, r, t)), [0.0, 0, 0])
                       for r in xrange(s + 1, t + 1)]
                      +
                      [([c.Node(TriSkipped, Right, s, t)],
                        [0, 0, 0])])

        c.set(c.Node(t, Right, 0, n),
              [([c.Node(t, Right, 0, n-1)], [0,0,0])
               for t in [Tri, TriSkipped]])


    def parse_second_bigram(self, sent_len, scorer, chart=None,
                            single_root=True):
        n = sent_len + 1
        c = chart

        self._initialize(c, n, scorer)

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break

                if s != 0:
                    c.set(c.Node(Box, Left, s, t),
                          (([c.Node(Tri, Right, s, r),
                             c.Node(Tri, Left, r+1, t)],
                            [0,0,0])
                           for r in range(s, t)))

                if s != 0:
                    c.set(c.Node(Trap, Left, s, t),
                          [([c.Node(Tri, Right, s, t-1),
                             c.Node(Tri, Left, t, t)],
                            [scorer.arc_score(t, s, t), 1, 0])] +
                          [([c.Node(Box, Left, s, r),
                             c.Node(Trap, Left, r, t)],
                            [scorer.arc_score(t, s, r), 1, 0])
                           for r in range(s+1, t)])

                boxes = []
                if s != 0 or not single_root:
                    boxes = \
                      [([c.Node(Trap, Right, s, r),
                         c.Node(Box, Left, r, t)],
                        [scorer.arc_score(s, t, r), 1, 0])
                       for r  in range(s + 1, t)]

                c.set(c.Node(Trap, Right, s, t),
                      [([c.Node(TriSkipped, Right, s, r),
                         c.Node(Tri, Left, r+1, t)],
                        [scorer.arc_score(s, t, s), 1, 0])
                       for r in range(s, t)] + boxes)

                if s != 0:
                    c.set(c.Node(Tri, Left, s, t),
                          (([c.Node(Tri, Left, s, r),
                             c.Node(Trap, Left, r, t)],
                            [0, 0 , 0])
                           for r  in range(s, t)))

                c.set(c.Node(TriSkipped, Right, s, t),
                      [([c.Node(TriSkipped1, Right, s, s)],
                        [scorer.bigram_score(s, t+1, True), 0, t - s])])


                c.set(c.Node(Tri, Right, s, t),
                      ([([c.Node(Trap, Right, s, r),
                          c.Node(Tri, Right, r, t)], [0, 0, 0])
                        for r in range(s + 1, t + 1)] +
                      [([c.Node(TriSkipped, Right, s, t)], [0,0,0])]))

        c.set(c.Node(t, Right, 0, n),
              [([c.Node(t, Right, 0, n-1)], [0,0,0])
               for t in [Tri, TriSkipped]])
