from skipdep.parse import Parse, Scorer
import skipdep.interface_pydecode as ip
import time
import random

def dev_speed(mode="INTER"):
    for l in list(open("sizes")):
        n, m = map(int, l.split())
        arc_scores = [[random.random() -0.75 for i in range(n+1)]
                      for j in range(n+1)]
        bigram_scores = [[random.random()-0.75 for i in range(n+3)]
                         for j in range(n+3)]
        score = Scorer(n, arc_scores, bigram_scores=bigram_scores)

        tim = time.time()
        ip.parse_bigram(n, score, m)
        print "DONE A", time.time() - tim

        tim = time.time()
        b = ip.Bisector(min_val=-10, max_val=10, limit=20)
        ip.parse_binary_search(n, score, m, b)
        print "DONE B", time.time() - tim

        tim = time.time()
        ip.parse_bigram(n, score, None)
        print "DONE C", time.time() - tim

        print b.history

if __name__ == "__main__":
    dev_speed()
