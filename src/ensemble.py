import numpy as np

def combine_scores(rnn_pred, news_score, pe_z, pct_from_high, weights=None):
    # rnn_pred, news_score: scalars or arrays standardized
    if weights is None:
        weights = dict(r=3.0, n=2.0, pe=1.0, h=-1.0)
    score = weights['r']*rnn_pred + weights['n']*news_score + weights['pe']*pe_z + weights['h']*pct_from_high
    return score