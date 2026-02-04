# relevance.py

def is_relevant(q, t):

    q = set(q.lower().split())
    t = set(t.lower().split())

    return len(q & t) >= 2
