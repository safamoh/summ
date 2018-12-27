from __future__ import division
import os
import re


"""
(text teaser)
EXTRA FEATURES
https://github.com/xiaoxu193/PyTeaser/blob/master/pyteaser.pyo
TextTeaser associates a score with every sentence. This score is a linear combination of 
features extracted from that sentence.

Relevance to the title
Relevance to keywords in the article
Position of the sentence
Length of the sentence

        

"""


def sentence_position(i, size):
    """different sentence positions indicate different
    probability of being an important sentence"""

    normalized = i*1.0 / size
    if 0 < normalized <= 0.1:
        return 0.17
    elif 0.1 < normalized <= 0.2:
        return 0.23
    elif 0.2 < normalized <= 0.3:
        return 0.14
    elif 0.3 < normalized <= 0.4:
        return 0.08
    elif 0.4 < normalized <= 0.5:
        return 0.05
    elif 0.5 < normalized <= 0.6:
        return 0.04
    elif 0.6 < normalized <= 0.7:
        return 0.06
    elif 0.7 < normalized <= 0.8:
        return 0.04
    elif 0.8 < normalized <= 0.9:
        return 0.04
    elif 0.9 < normalized <= 1.0:
        return 0.15
    else:
        return 0
        
        


if __name__=='__main__':
    #branches=['select_top_cos_sims']
    branches=[]

    for b in branches:
        globals()[b]()





































































