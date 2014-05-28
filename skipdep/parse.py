import itertools
from itertools import izip


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

    def check_single_root(self):
        return len([m for m in self.heads if m == 0]) <= 1

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

            if (not parse.check_projective()) or (not parse.check_spanning()) or (not parse.check_single_root()): continue
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

    def bigram_score(self, prev, current, with_penalty=False):
        """
        Returns
        -------
        score : float
           The score of prev->current
        """
        score = self._bigram_scores[prev][current]
        if with_penalty:
            score += ((current - prev - 1) * self.skip_penalty)
        return score


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
