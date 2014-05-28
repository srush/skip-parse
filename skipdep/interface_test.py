from itertools import izip
from skipdep.parse import Parse, Scorer
import nose.tools as nt
import interface_pydecode as ip

import random
def test_parse():
    parse = Parse([-1, 3, 3, 0, 3, 3])
    print parse.siblings(1)
    assert(parse.siblings(1) == [2,4,5])
    assert(parse.sibling(1) == 2)
    assert(parse.sibling(2) == 3)
    assert(parse.sibling(4) == 3)
    assert(parse.sibling(5) == 4)

    parse = Parse([-1, 3, 1, 0, 5, 3])
    assert(parse.siblings(1) == [5])
    assert(parse.sibling(1) == 3)
    assert(parse.sibling(5) == 3)


    parse = Parse([-1, 3, None, 4, 0])
    assert(parse.siblings(1) == [])
    assert(parse.sibling(1) == 3)


    parse = Parse([-1, None, None, None])
    assert(parse.check_spanning())
    assert(parse.check_projective())

    print  [p.heads for p in Parse.enumerate_projective(3, None)]
    assert(parse in Parse.enumerate_projective(3, None))

def test_parsing():
    for i in range(1):
        for n in range (2, 5):
            for m in range(1, n+1):

                arc_scores = [[random.random()-0.75
                               for i in range(n+1)]
                              for j in range(n+1)]

                second_order_scores = [[[random.random()-0.75
                                         for i in range(n+1)]
                                         for k in range(n+1)]
                                       for j in range(n+1)]

                bigram_scores = [[random.random()-0.75
                                  for i in range(n+3)]
                                 for j in range(n+3)]

                score2 = Scorer(n, arc_scores, bigram_scores)
                score4 = Scorer(n, arc_scores, bigram_scores,
                                skip_penalty=random.random(),
                                second_order=second_order_scores)

                yield check_bigram_ip, n, m, score2
                yield check_bigram_second_any, n, m, score4
                yield check_binary_search, n, m, score2

def parse_all(n, scorer, m=None):
    return max(Parse.enumerate_projective(n, m),
               key = scorer.score)

def check_bigram_ip(n, m, scorer):
    parse = parse_all(n, scorer, m)
    parse2 = ip.parse_bigram(n, scorer, m)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)
    assert(parse == parse2)

def check_bigram_second_any(n, m, scorer):
    parse = parse_all(n, scorer, m)
    parse2 = ip.parse_second_bigram(n, scorer, m)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)
    print parse.skipped_words(), parse2.skipped_words()
    assert(parse == parse2)

def check_binary_search(n, m, scorer):
    parse = parse_all(n, scorer, m)
    bisector = ip.Bisector()
    parse2 = ip.parse_binary_search(n, scorer, m, bisector)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)
    assert(parse == parse2)
