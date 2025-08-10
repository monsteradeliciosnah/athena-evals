from __future__ import annotations
import math, re
from collections import Counter
from typing import List, Dict

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def f1_score(pred: str, gold: str) -> float:
    p = normalize(pred).split(); g = normalize(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2*precision*recall/(precision+recall)

def rouge_l(pred: str, gold: str) -> float:
    # simple LCS-based approximation
    p = normalize(pred).split(); g = normalize(gold).split()
    dp = [[0]*(len(g)+1) for _ in range(len(p)+1)]
    for i in range(1, len(p)+1):
        for j in range(1, len(g)+1):
            dp[i][j] = dp[i-1][j-1]+1 if p[i-1]==g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    if not p or not g: return 0.0
    prec = lcs/len(p); rec = lcs/len(g)
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)
