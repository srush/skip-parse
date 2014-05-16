from itertools import izip
from compression.interface import Parser, Parse, Scorer
import nose.tools as nt

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

                arc_scores = [[random.random() -0.75 for i in range(n+1) ] for j in range(n+1)]
                second_order_scores = [[ [random.random() -0.75 for i in range(n+1) ] for k in range(n+1)] for j in range(n+1)]
                second_order_zero_scores = [[ [0.0 for i in range(n+1) ] for k in range(n+1)] for j in range(n+1)]
                bigram_scores = [[random.random() -0.75 for i in range(n+3) ] for j in range(n+3)]
                bigram_zero_scores = [[0.0 for i in range(n+3) ] for j in range(n+3)]
                score1 = Scorer(n, arc_scores)
                score2 = Scorer(n, arc_scores, bigram_scores)
                score3 = Scorer(n, arc_scores, bigram_scores, skip_penalty=random.random())
                # score4 = Scorer(n, arc_scores, bigram_scores, skip_penalty=random.random(), second_order=second_order_scores)
                score4 = Scorer(n, arc_scores, bigram_scores, skip_penalty=random.random(), second_order=second_order_scores)
                yield check_skip, n, m, score1
                yield check_bigram, n, m, score2
                yield check_bigram, n, None, score2
                yield check_bigram, n, None, score3
                # yield check_binary_search, n, m, score2
                yield check_bigram_second, n, None, score4

def parse_all(n, scorer, m=None):
    return max(Parse.enumerate_projective(n, m),
               key = scorer.score)

def check_skip(n, m, scorer):
    parse = parse_all(n, scorer, m)
    parse2 = Parser().parse_skip(n, scorer, m)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)
    assert(parse == parse2)

def check_bigram(n, m, scorer):
    parse = parse_all(n, scorer, m)
    parse2 = Parser().parse_bigram(n, scorer, m)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)
    assert(parse == parse2)

def check_bigram_second(n, m, scorer):
    parse = parse_all(n, scorer, m)
    parse2 = Parser().parse_second_bigram(n, scorer, m)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)
    print parse.skipped_words(), parse2.skipped_words()
    assert(parse == parse2)


def check_binary_search(n, m, scorer):
    parse = parse_all(n, scorer, m)
    parse2 = Parser().parse_binary_search(n, scorer, m)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)

    assert(parse == parse2)


def biggie():
    n = 30
    m = 20
    arc_scores = [[random.random() -0.75 for i in range(n+1) ] for j in range(n+1)]
    bigram_scores = [[random.random() -0.75 for i in range(n+3) ] for j in range(n+3)]
    score1 = Scorer(n, arc_scores)
    score2 = Scorer(n, arc_scores, bigram_scores)
    # Parser().parse_skip(n, score1, m)
    # Parser().parse_bigram(n, score2, m)
    Parser().parse_binary_search(n, score2, m)

def pydecode():
    import interface_pydecode as ip
    n = 50
    m = 20
    arc_scores = [[random.random() -0.75 for i in range(n+1) ] for j in range(n+1)]
    bigram_scores = [[random.random() -0.75 for i in range(n+3) ] for j in range(n+3)]
    score1 = Scorer(n, arc_scores)
    score2 = Scorer(n, arc_scores, bigram_scores)
    # Parser().parse_skip(n, score1, m)
    # Parser().parse_bigram(n, score2, m)
    ip.parse_binary_search(n, score2, m)

if __name__ == "__main__":
    pydecode()
