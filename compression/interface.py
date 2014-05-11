from itertools import izip
import itertools
from collections import namedtuple, defaultdict
import pydecode.hyper as ph

# range = xrange
class Parse:
    """
    Class representing a dependency parse with
    possible unused modifier words.

    """

    def __init__(self, heads):
        """
        Parameters
        ----------
        head : List
           The head index of each modifier or None for unused modifiers.
           Requires head[0] = -1 for convention.
        """
        self.heads = heads
        assert(self.heads[0] == -1)

    def __eq__(self, other):
        return self.heads == other.heads

    def arcs(self, second=False):
        """
        Returns
        -------
        arc : iterator of (m, h) pairs in m order
           Each of the arcs used in the parse.
        """
        for m, h in enumerate(self.heads):
            if m == 0 or h is None: continue
            yield (m, h)

    def siblings(self, m):
        return [m2 for (m2, h) in self.arcs()
                if h == self.heads[m]
                if m != m2]

    def sibling(self, m):
        sibs = self.siblings(m)
        h = self.heads[m]
        if m > h:
            return max([s2 for s2 in sibs if h < s2 < m] + [h])
        if m < h:
            return min([s2 for s2 in sibs if h > s2 > m] + [h])


    def sequence(self):
        """
        Returns
        -------
        sequence : iterator of m indices in m order
           Each of the words used in the sentence,
           by convention starts with 0 and ends with n+1.
        """
        yield 0
        for m, h in self.arcs():
            yield m
        yield len(self.heads)

    def skipped_words(self):
        return len([h for h in self.heads if h is None])

    def check_spanning(self):
        """
        Is the parse tree as valid spanning tree?

        Returns
        --------
        spanning : bool
           True if a valid spanning tree.
        """
        d = {}
        for m, h in self.arcs():
            if m == h:
                return False

            d.setdefault(h, [])
            d[h].append(m)
        stack = [0]
        seen = set()
        while stack:
            cur = stack[0]
            if cur in seen:
                return False
            seen.add(cur)
            stack = d.get(cur,[]) + stack[1:]
        if len(seen) != len(self.heads) - len([1 for p in self.heads if p is None]):
            return False
        return True

    def check_projective(self):
        """
        Is the parse tree projective?

        Returns
        --------
        projective : bool
           True if a projective tree.
        """

        for m, h in self.arcs():
            for m2, h2 in self.arcs():
                if m2 == m: continue
                if m < h:
                    if m < m2 < h < h2 or m < h2 < h < m2 or \
                            m2 < m < h2 < h or  h2 < m < m2 < h:
                        return False
                if h < m:
                    if h < m2 < m < h2 or h < h2 < m < m2 or \
                            m2 < h < h2 < m or  h2 < h < m2 < m:
                        return False
        return True

    @staticmethod
    def enumerate_projective(n, m=None):
        """
        Enumerate all possible projective trees.

        Parameters
        ----------
        n - int
           Length of sentence (without root symbol)

        m - int
           Number of modifiers to use.

        Returns
        --------
        parses : iterator
           An iterator of possible (m,n) parse trees.
        """
        for mid in itertools.product([None] + range( n + 1), repeat=n):
            parse = Parse([-1] + list(mid))
            if m is not None and parse.skipped_words() != n- m:
                continue

            if (not parse.check_projective()) or (not parse.check_spanning()): continue
            yield parse


class Scorer(object):
    """
    Object for scoring parse structures.
    """

    def __init__(self, n, arc_scores,
                 bigram_scores=None,
                 skip_penalty=0.0,
                 second_order=None):
        """
        Parameters
        ----------
        n : int
           Length of the sentence (without root).

        arc_scores : 2D array (n+1 x n+1)
           Scores for each possible arc
           arc_scores[h][m].

        bigram_scores : 2D array (n+2 x n+2)
           Scores for each possible modifier bigram
           bigram_scores[prev][cur].

        second_order : 3d array (n+2 x n+2 x n+2)
           Scores for each possible modifier bigram
           second_order[h][s][m].
        """
        self._arc_scores = arc_scores
        self._second_order_scores = second_order
        self._bigram_scores = bigram_scores
        self.skip_penalty = skip_penalty

    def arc_score(self, head, modifier, sibling=None):
        """
        Returns
        -------
        score : float
           The score of head->modifier
        """
        assert((sibling is None) or (self._second_order_scores is not None))
        if sibling is None:
            return self._arc_scores[head][modifier]
        else:
            return self._arc_scores[head][modifier] + \
                self._second_order_scores[head][sibling][modifier]

    def bigram_score(self, prev, current):
        """
        Returns
        -------
        score : float
           The score of prev->current
        """
        return self._bigram_scores[prev][current]

    def score(self, parse):
        """
        Score a parse based on arc score and bigram score.

        Parameters
        ----------
        parse : Parse
            The parse to score.

        Returns
        -------
        score : float
           The score of the parse structure.
        """
        parse_score = 0.0
        if self._second_order_scores is None:
            parse_score = \
                sum((self.arc_score(h, m)
                     for m, h in parse.arcs()))
        else:
            parse_score = \
                sum((self.arc_score(h, m, parse.sibling(m))
                     for m, h in parse.arcs()))



        bigram_score = 0.0
        if self._bigram_scores is not None:
            seq = list(parse.sequence())
            bigram_score = sum((self.bigram_score(i, j)
                                for i, j in izip(seq, seq[1:])))

        bigram_score -= parse.skipped_words() * self.skip_penalty
        return parse_score + bigram_score

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


# Globals
Tri = 1
TrapSkipped = 2
Trap = 3
Box = 4

Right = 0
Left = 1

def Edge(a, b=0.0):
    return (a, b)
# class Edge(namedtuple("Edge", ["tail", "score"])):
#     def __new__(cls, tail, score=0.0):
#         return super(Edge, cls).__new__(cls, tail, score)

def NodeType(type, dir, span, count, states=None) :
    if states is None:
        return (type, dir, span, count)
    return (type, dir, span, count, states)
def node_type(nodetype): return nodetype[0]
def node_span(nodetype): return nodetype[2]
def node_dir(nodetype): return nodetype[1]

class Arc(namedtuple("Arc", ["head", "mod", "sibling"])):
    def __new__(cls, head, mod, sibling=None):
        return super(Arc, cls).__new__(cls, head, mod, sibling)


class Bigram(namedtuple("Bigram", ["s1", "s2"])):
    pass

# class NodeType(namedtuple("NodeType", ["type", "dir", "span", "count", "states"])):
#     """
#     Representation of an iterm in the parse chart.

#     Attributes
#     ----------
#     type : {trap, tri, trapskipped}

#     dir : {left, right}

#     span : pair of ints

#     count : int
#        How many arcs are under this item?

#     states : pair
#        What is the FSA state before and after this item?

#     """
#     Names = {0 : "right",
#              1 : "left"}

#     Type = {0 : "tri",
#             1 : "trapskipped",
#             2 : "trap"}

#     def __new__(cls, type, dir, span, count, states=None):
#         return super(NodeType, cls).__new__(cls, type, dir, span, count, states)

#     def __str__(self):
#         return "%s %s %d-%d %d %s"%(NodeType.Type[self.type], NodeType.Dir[self.dir],
#                                     self.span[0], self.span[1], self.count, self.states)


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
            if span[0] == span[1]:
                print element
        if isinstance(element, tuple) and  node_type(element) == Trap:
            span = node_span(element)

            if node_dir(element) == Right:
                mods[span[1]] = span[0]
            if node_dir(element) == Left:
                mods[span[0]] = span[1]
    return Parse(mods)

class Parser(object):
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
        [c.initialize(NodeType(sh, d, (s, s), 0), 0.0)
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
                        c.set(NodeType(Trap, Left, span, mod_count),
                              [Edge([key1, key2], Arc(t, s))
                               for r in range(s, t)
                               for m1 in range(mod_count)
                               for key1 in [(Tri, Right, (s, r), m1)]
                               if key1 in c.chart
                               for m2 in [mod_count  - m1 - 1]
                               for key2 in [(Tri, Left, (r+1, t), m2)]
                               if key2 in c.chart
                               ])

                    if mod_count > 0:
                        c.set(NodeType(Trap, Right, span, mod_count),
                              [Edge([key1, key2], Arc(s, t))
                               for r in range(s, t)
                               for m1 in range(mod_count)
                               for key1 in [(Tri, Right, (s, r), m1)]
                               if key1 in c.chart
                               for m2 in [mod_count - m1 - 1]
                               for key2 in [(Tri, Left, (r+1, t), m2)]
                               if key2 in c.chart
                               ])


                    c.set(NodeType(TrapSkipped, Right, span, mod_count),
                          [Edge([key1, key2])
                           for m1 in range(mod_count + 1)
                           for key1 in [(Tri, Right, (s, t-1), m1)]
                           if key1 in c.chart
                           for m2 in [mod_count - m1]
                           for key2 in [(Tri, Left, (t, t), m2)]
                           if key2 in c.chart
                           ])

                    if s != 0:
                        c.set(NodeType(Tri, Left, span, mod_count),
                              [Edge([key1, key2])
                               for r in range(s, t)
                               for m1 in range(mod_count + 1 )
                               for key1 in [(Tri, Left, (s, r), m1)]
                               if key1 in c.chart
                               for m2 in [mod_count - m1]
                               for key2 in [(Trap, Left, (r, t), m2)]
                               if key2 in c.chart])

                    c.set(NodeType(Tri, Right, span, mod_count),
                          [Edge([key1, key2])
                           for r in range(s + 1, t + 1)
                           for m1 in range(mod_count + 1)
                           for key1 in [(Trap, Right, (s, r), m1)]
                           if key1 in c.chart
                           for m2 in [mod_count - m1]
                           for key2 in [(Tri, Right, (r, t), m2)]
                           if key2 in c.chart] + \
                          [Edge([key1, key2])
                           for m1 in range(mod_count + 1)
                           for key1 in [(TrapSkipped, Right, (s, t), m1)]
                           if key1 in c.chart
                           for m2 in [mod_count  - m1]
                           for key2 in [(Tri, Right, (t, t), m2)]
                           if key2 in c.chart
                           ])

        return make_parse(n, c.backtrace(NodeType(Tri, Right, (0, n-1), m)))


    # def parse_bigram(self, sent_len, scorer, m):
    #     """
    #     Parses with bigrams.

    #     Parameters
    #     -----------
    #     n : int
    #        The length of the sentence.

    #     scorer : Scorer
    #        The arc-factored weights and bigram scores.

    #     m : int
    #        The length of the compressed sentence.
    #        None if any length allowed.

    #     Returns
    #     -------
    #     parse : Parse
    #        The best dependency parse with these constraints.
    #     """
    #     n = sent_len + 1

    #     c = Chart()

    #     diff = n - m

    #     # Initialize the chart.
    #     [c.initialize(NodeType(sh, d, (s, s), 0, (s1, s2)),
    #                   scorer.bigram_score(s1, s2) if s1 != s2 else 0.0)
    #      for s in range(n)
    #      for d in [Right, Left]
    #      for sh in [Trap, Tri]
    #      for s1 in (range(s-diff, s + 1) if d == Left else [s])
    #      for s2 in [s]]

    #     for k in range(1, n):
    #         for s in range(n):
    #             t = k + s
    #             if t >= n: break
    #             span = (s, t)
    #             remaining = n - s
    #             for mod_count in range(m - remaining -1, m + 1):
    #                 for s1 in range(s-diff, s+1):
    #                     # First create incomplete items.
    #                     if s != 0 and mod_count > 0:
    #                         c.set(NodeType(Trap, Left, span, mod_count, (s1, t)),
    #                               (Edge([key1, key2], scorer.arc_score(t, s))
    #                                for r  in range(s, t)
    #                                for m1 in range(mod_count)
    #                                for s2 in range(s, r + 2)
    #                                for key1 in [(Tri, Right, (s, r), m1, (s1, s2))]
    #                                if key1 in c.chart
    #                                for m2 in [mod_count - m1 - 1]
    #                                for key2 in [(Tri, Left, (r+1, t), m2, (s2, t))]
    #                                if key2 in c.chart))


    #                     if mod_count > 0:
    #                         c.set(NodeType(Trap, Right, span, mod_count, (s1, t)),
    #                               (Edge([key1, key2], scorer.arc_score(s, t))
    #                                for r  in range(s, t)
    #                                for s2 in range(s, r + 2)
    #                                for m1 in range(mod_count)
    #                                for key1 in [(Tri, Right, (s, r), m1, (s1, s2))]
    #                                if key1 in c.chart
    #                                for m2 in [mod_count - m1 - 1]
    #                                for key2 in [(Tri, Left, (r+1, t), m2, (s2, t))]
    #                                if key2 in c.chart))


    #                     for s3 in range(s1, t+1):
    #                         c.set(NodeType(TrapSkipped, Right, span, mod_count, (s1, s3)),
    #                               (Edge([key1, key2], 0.0)
    #                                for m1 in range(mod_count + 1)
    #                                for key1 in [(Tri, Right, (s, t-1), m1, (s1, s3))]
    #                                if key1 in c.chart
    #                                for m2 in [mod_count - m1]
    #                                for key2 in [(Tri, Left, (t, t), m2, (t, t))]
    #                                if key2 in c.chart))

    #                 for s1 in range(s-diff, s+1):
    #                     for s3 in range(s1, t+1):
    #                         if s != 0:
    #                             c.set(NodeType(Tri, Left, span, mod_count, (s1, s3)),
    #                                   (Edge([key1, key2])
    #                                    for r  in range(s, t)
    #                                    for m1 in range(mod_count + 1)
    #                                    for key1 in [(Tri, Left, (s, r), m1, (s1, r))]
    #                                    if key1 in c.chart
    #                                    for m2 in [mod_count - m1]
    #                                    for key2 in [(Trap, Left, (r, t), m2, (r, s3))]
    #                                    if key2 in c.chart))

    #                         c.set(NodeType(Tri, Right, span, mod_count, (s1, s3)),
    #                               itertools.chain((Edge([key1, key2])
    #                                                for r in range(s + 1, t + 1)
    #                                                for m1 in range(mod_count + 1)
    #                                                for key1 in [(Trap, Right, (s, r), m1, (s1, r))]
    #                                                if key1 in c.chart
    #                                                for m2 in [mod_count - m1]
    #                                                for key2 in [(Tri, Right, (r, t), m2, (r, s3))]
    #                                                if key2 in c.chart),

    #                                               (Edge([key1,key2])
    #                                                for m1 in range(mod_count + 1)
    #                                                for key1 in [(TrapSkipped, Right, (s, t), m1, (s1, s3))]
    #                                                if key1 in c.chart
    #                                                for m2 in [mod_count  - m1]
    #                                                for key2 in [(Tri, Right, (t, t), m2, (t, t))]
    #                                                if key2 in c.chart)))

    #     c.set(NodeType(Tri, Right, (0, n-1), m, (0, n)),
    #           [Edge([key1], scorer.bigram_score(s3, n))
    #            for s3 in range(n)
    #            for key1 in [(Tri, Right, (0, n-1), m, (0, s3))]
    #            if key1 in c.chart])
    #     return make_parse(n, c.backtrace(NodeType(Tri, Right, (0, n-1), m, (0, n))))

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

        if chart != None:
            c = chart
        else:
            c = Chart(score)
        c.score = score

        diff = n
        any_size = (m is None)
        if m is not None:
            diff = n - m


        # Initialize the chart.
        tmp = 0
        for s in range(n):
             for d in [Right, Left]:
                 for sh in [Trap, Tri]:
                     for s1 in [s]:
                         for s2 in (range(s,  n + 1) if d == Right else [s]):
                             tmp += 1
                             c.initialize(tmp)#NodeType(sh, d, (s, s), 0, (s1, s2)))
                             c.set(NodeType(sh, d, (s, s), 0, (s1, s2)),
                                   [Edge([tmp], Bigram(s1, s2))])


        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)
                remaining = n - s

                if any_size:
                    mod_counts = [0]
                else:
                    mod_counts = range(m - remaining - 1, m + 1)

                for mod_count in mod_counts:
                    #for s1 in range(s-diff, s+1):
                    for s3 in range(t, min(n+1, t + diff + 1)):
                        # First create incomplete items.
                        if s != 0 and (mod_count > 0 or any_size):
                            c.set(NodeType(Trap, Left, span, mod_count, (s, s3)),
                                  (Edge([key1, key2], Arc(t, s))
                                   for r  in range(s, t)
                                   for m1 in (range(mod_count) if not any_size else [0])
                                   for key1 in [(Tri, Right, (s, r), m1, (s, r+1))]
                                   if key1 in c.chart
                                   for m2 in ([mod_count - m1 - 1] if not any_size else [0])
                                   for key2 in [(Tri, Left, (r+1, t), m2, (r+1, s3))]
                                   if key2 in c.chart))


                        if mod_count > 0 or any_size:
                            c.set(NodeType(Trap, Right, span, mod_count, (s, s3)),
                                  (Edge([key1, key2], Arc(s, t))
                                   for r  in range(s, t)
                                   for m1 in (range(mod_count) if not any_size else [0])
                                   for key1 in [(Tri, Right, (s, r), m1, (s, r+1))]
                                   if key1 in c.chart
                                   for m2 in ([mod_count - m1 - 1] if not any_size else [0])
                                   for key2 in [(Tri, Left, (r+1, t), m2, (r+1, s3))]
                                   if key2 in c.chart))

                        c.set(NodeType(TrapSkipped, Right, span, mod_count, (s, s3)),
                                  (Edge([key1, key2], 0.0)
                                   for m1 in (range(mod_count + 1) if not any_size else [0])
                                   for key1 in [(Tri, Right, (s, t-1), m1, (s, s3))]
                                   if key1 in c.chart
                                   for m2 in ([mod_count - m1] if not any_size else [0])
                                   for key2 in [(Tri, Left, (t, t), m2, (t, t))]
                                   if key2 in c.chart))

                    #for s1 in range(s-diff, s+1):
                    for s3 in range(t, min(n+1, t + diff + 1)):
                            if s != 0:
                                c.set(NodeType(Tri, Left, span, mod_count, (s, s3)),
                                      (Edge([key1, key2])
                                       for r  in range(s, t)
                                       for m1 in (range(mod_count + 1) if not any_size else [0])
                                       for key1 in [(Tri, Left, (s, r), m1, (s, r))]
                                       if key1 in c.chart
                                       for m2 in ([mod_count - m1] if not any_size else [0])
                                       for key2 in [(Trap, Left, (r, t), m2, (r, s3))]
                                       if key2 in c.chart))

                            c.set(NodeType(Tri, Right, span, mod_count, (s, s3)),
                                  itertools.chain((Edge([key1, key2])
                                                   for r in range(s + 1, t + 1)
                                                   for m1 in (range(mod_count + 1) if not any_size else [0])
                                                   for key1 in [(Trap, Right, (s, r), m1, (s, r))]
                                                   if key1 in c.chart
                                                   for m2 in ([mod_count - m1] if not any_size else [0])
                                                   for key2 in [(Tri, Right, (r, t), m2, (r, s3))]
                                                   if key2 in c.chart),

                                                  (Edge([key1,key2])
                                                   for m1 in (range(mod_count + 1) if not any_size else [0])
                                                   for key1 in [(TrapSkipped, Right, (s, t), m1, (s, s3))]
                                                   if key1 in c.chart
                                                   for m2 in ([mod_count  - m1] if not any_size else [0])
                                                   for key2 in [(Tri, Right, (t, t), m2, (t, t))]
                                                   if key2 in c.chart)))

        return make_parse(n, c.backtrace(NodeType(Tri, Right, (0, n-1), m if not any_size else 0, (0, n))))


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
                # assert(scorer.arc_score(arc.head, arc.mod, arc.sibling) !=  scorer.arc_score(arc.head, arc.mod))
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
        tmp = 0
        for s in range(n):
             for d in [Right, Left]:
                 for sh in [Tri]:
                     for s2 in (range(s,  n + 1) if d == Right else [s]):
                         tmp += 1
                         c.initialize("A" + str(tmp))
                         c.set(NodeType(sh, d, (s, s), 0, (0, s2)),
                               [Edge(["A" + str(tmp)], Bigram(s, s2))])


        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)

                for s3 in range(t, n+1):
                    if s != 0:
                        c.set(NodeType(Box, Left, span, 0, (1, s3)),
                              (Edge([key1, key2])
                               for r  in range(s, t)
                               for used in [0, 1]
                               for key1 in [(Tri, Right, (s, r), 0, (used, r+1))]
                               if key1 in c.chart
                               for used2 in [0, 1]
                               for key2 in [(Tri, Left, (r+1, t), 0, (used2, s3))]
                               if key2 in c.chart))

                    if s != 0:
                        c.set(NodeType(Trap, Left, span, 0, (1, s3)),
                              [Edge([key1, key2], Arc(t, s, t))
                               for used in [0, 1]
                               for key1 in [(Tri, Right, (s, t-1), 0, (used, t))]
                               if key1 in c.chart
                               for key2 in [(Tri, Left, (t, t), 0, (0, s3))]
                               if key2 in c.chart] +
                              [Edge([key1, key2], Arc(t, s, r))
                               for r in range(s+1, t)
                               for key1 in [(Box, Left, (s, r), 0, (1, r))]
                               if key1 in c.chart
                               for key2 in [(Trap, Left, (r, t), 0, (1, s3))]
                               if key2 in c.chart])

                    c.set(NodeType(Trap, Right, span, 0, (1, s3)),
                          [Edge([key1, key2], Arc(s, t, s))
                           for r in range(s, t)
                           for key1 in [(Tri, Right, (s, r), 0, (0, r+1))]
                           if key1 in c.chart
                           for used in [0, 1]
                           for key2 in [(Tri, Left, (r+1, t), 0, (used, s3))]
                           if key2 in c.chart] +
                          [Edge([key1, key2], Arc(s, t, r))
                           for r  in range(s + 1, t)
                           for key1 in [(Trap, Right, (s, r), 0, (1, r))]
                           if key1 in c.chart
                           for key2 in [(Box, Left, (r, t), 0, (1, s3))]
                           if key2 in c.chart])

                    for used in [0, 1]:
                        c.set(NodeType(TrapSkipped, Right, span, 0, (used, s3)),
                              (Edge([key1, key2], 0.0)
                               for key1 in [(Tri, Right, (s, t-1), 0, (used, s3))]
                               if key1 in c.chart
                               for key2 in [(Tri, Left, (t, t), 0, (0, t))]
                               if key2 in c.chart))

                # for s3 in range(t, n+1):
                #     c.set(NodeType(Trap, Right, span, 0, (s, s3)),
                #           (Edge([key1, key2], Arc(s, t, r))
                #            for r  in range(s, t)
                #            for key1 in [(Trap, Right, (s, r), 0, (s, r))]
                #            if key1 in c.chart
                #            for key2 in [(Box, Left, (r, t), 0, (r, s3))]
                #            if key2 in c.chart))

                #     c.set(NodeType(Trap, Left, span, 0, (s, s3)),
                #           (Edge([key1, key2], Arc(t, s, r))
                #            for r  in range(s+1, t)
                #            for key1 in [(Box, Left, (s, r), 0, (s, r))]
                #            if key1 in c.chart
                #            for key2 in [(Trap, Left, (r, t), 0, (r, s3))]
                #            if key2 in c.chart))


                for s3 in range(t, n+1):
                        if s != 0:
                            c.set(NodeType(Tri, Left, span, 0, (1, s3)),
                                  (Edge([key1, key2])
                                   for r  in range(s, t)
                                   for used in [0, 1]
                                   for key1 in [(Tri, Left, (s, r), 0, (used, r))]
                                   if key1 in c.chart
                                   for key2 in [(Trap, Left, (r, t), 0, (1, s3))]
                                   if key2 in c.chart))

                        for used in [0, 1]:
                            c.set(NodeType(Tri, Right, span, 0, (used, s3)),
                                  ([Edge([key1, key2])
                                   for r in range(s + 1, t + 1)
                                   for key1 in [(Trap, Right, (s, r), 0, (1, r))]
                                   if key1 in c.chart
                                   for used2 in [0, 1]
                                   for key2 in [(Tri, Right, (r, t), 0, (used2, s3))]
                                   if key2 in c.chart] if used == 1 else []) +

                                  [Edge([key1,key2])
                                   for key1 in [(TrapSkipped, Right, (s, t), 0, (used, s3))]
                                   if key1 in c.chart
                                   for key2 in [(Tri, Right, (t, t), 0, (0, t))]
                                   if key2 in c.chart])

        # print c.chart[NodeType(Tri, Right, (0, n-1), 0, (0, n))]
        c.set("final",
              [Edge([NodeType(Tri, Right, (0, n-1), 0, (used, n))])
               for used in [0, 1]])
        print c.chart["final"]
        return make_parse(n, c.backtrace("final"))
