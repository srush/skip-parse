from itertools import izip
from compression.interface import Parser, Parse, Scorer
import nose.tools as nt

import random

def test_parsing():
    for i in range(1):
        for n in range (2, 5):
            for m in range(1, n+1):

                arc_scores = [[random.random() -0.75 for i in range(n+1) ] for j in range(n+1)]
                bigram_scores = [[random.random() -0.75 for i in range(n+3) ] for j in range(n+3)]
                score1 = Scorer(n, arc_scores)
                score2 = Scorer(n, arc_scores, bigram_scores)
                score3 = Scorer(n, arc_scores, bigram_scores, skip_penalty=random.random())
                yield check_skip, n, m, score1
                yield check_bigram, n, m, score2
                yield check_bigram, n, None, score2
                yield check_bigram, n, None, score3
                yield check_binary_search, n, m, score2

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

def check_binary_search(n, m, scorer):
    parse = parse_all(n, scorer, m)
    parse2 = Parser().parse_binary_search(n, scorer, m)
    print parse.heads, scorer.score(parse)
    print parse2.heads, scorer.score(parse2)
    assert(parse == parse2)


def test_biggie():
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
    n = 30
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
