from __future__ import division
import nltk.collocations
import nltk.corpus
import collections
from math import log
from scipy.stats import binom

from duc_reader import files2sentences


#0v1# JC  Nov 20, 2018  Setup base likelihood calculation


def load_entire_corpus():
    documents,sentences,sentences_topics=files2sentences(limit_topic='')
    all_corpus=" ".join(sentences)
    word_tokens = nltk.wordpunct_tokenize(" ".join(sentences).lower())
    print ("[verbose] using "+str(len(word_tokens))+" words.")
    return word_tokens


def bigram_counts():
    #Likehood on bigrams allows for next term prediction
    bgm = nltk.collocations.BigramAssocMeasures() #<-- likelihood a part

    finder = nltk.collocations.BigramCollocationFinder.from_words(load_entire_corpus())
    scored = finder.score_ngrams(bgm.likelihood_ratio)

    # Group bigrams by first word in bigram.                                        
    prefix_keys = collections.defaultdict(list)
    for key, scores in scored:
        prefix_keys[key[0]].append((key[1], scores))
    
    for key in prefix_keys:
        prefix_keys[key].sort(key = lambda x: -x[1])
    
    c=0
    for key in prefix_keys:
        c+=1
        print ("OK: "+str(key)+"--> "+str(prefix_keys[key]))
        if c>10:break

   # prefix_keys['baseball']
    return

def word_counter(words):
    wcount={}
    for word in words:
        try:wcount[word]+=1
        except:wcount[word]=1
    return wcount,len(words)

def loglikefun(document,tdm_count,total_words):
    doc_words=nltk.wordpunct_tokenize(document)
    doc_count_words,doc_word_length=word_counter(doc_words)
    
    #Binomial distribution
    #dbinom(x, size, prob, log = FALSE)
    #dbinom(3, 5, 0.8)
    #the equivalent of dbinom(), use scipy.stats.binom.pmf(3,5,0.8).
    #binom.pmf(k, n, p, loc) 
    
    ratios={}
    for word in set(doc_words):
        i_words=doc_count_words[word]
        i_size=doc_word_length
        
        #L1 <- dbinom(I.words, size = sum(I.words), prob = I.words/sum(I.words))
        l1=binom.pmf(i_words,i_size,i_words/i_size)

        #L2 <- dbinom(B.words, size = sum(B.words), prob = B.words/sum(B.words))
        b_words=tdm_count[word]
        b_size=total_words
        l2=binom.pmf(b_words,b_size,b_words/b_size)
    
        #L3 <- dbinom(I.words, size = sum(I.words), prob =  (I.words+B.words)/(sum(I.words)+sum(B.words)))
        prob=(i_words+b_words)/(i_size+b_size)
        l3=binom.pmf(i_words,i_size,prob)

        #L4 <- dbinom(B.words, size = sum(B.words), prob = (I.words+B.words)/(sum(I.words)+sum(B.words)) )
        prob=(i_words+b_words)/(i_size+b_size)
        l4=binom.pmf(b_words,b_size,prob)
        
        
        ratios[word]=2*(log(l1)+log(l2)-log(l3)-log(l4))
    
    return ratios


def loglikelyhood():
    #news.TDM <- TermDocumentMatrix(Doc.corpus, control = list(tokenize = BigramTokenizer)) 
    #news.TDM <- as.matrix(news.TDM)
    ##find all corpus total words freqancies
    #TotalWords <- rowSums(news.TDM)
    topic_signatures=[]

    cutoff=10.83
    
    branches=[]
    branches=['count_all_words']

    #1/  Count all words
    print ("Counting words...")
    if 'count_all_words' in branches:
        tdm_count,total_words=word_counter(load_entire_corpus())
    
    #2/  For each subset of document/topic
    documents,sentences,sentences_topics=files2sentences(limit_topic='')
    print ("Calculating log likelyhood...")
    for doc in documents:
        print ("YO: "+str(doc))
        ratios=loglikefun(doc,tdm_count,total_words)
        print ("[debug] ALL RATIOS: "+str(ratios))
        
        for word in ratios:
            if ratios[word]>cutoff: topic_signatures+=[word]
        
        print ("[debug] topic signatures: "+str(topic_signatures))
        break

    return

if __name__=='__main__':
    branches=['bigram_counts']
    branches=['loglikelyhood']
    for b in branches:
        globals()[b]()


"""
other samples:  https://www.programcreek.com/python/example/107928/nltk.metrics.BigramAssocMeasures.likelihood_ratio

nltk.metrics.association.NgramAssocMeasures.likelihood_ratio

"""
























