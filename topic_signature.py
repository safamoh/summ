from __future__ import division
import nltk.collocations
import nltk.corpus
import collections
from math import log
from scipy.stats import binom

from duc_reader import files2sentences


#0v1# JC  Nov 20, 2018  Setup base likelihood calculation


def load_entire_corpus_words():
    documents,sentences,sentences_topics=files2sentences(limit_topic='')
    all_corpus=" ".join(sentences)
    word_tokens = nltk.wordpunct_tokenize(" ".join(sentences).lower())
    print ("[verbose] using "+str(len(word_tokens))+" words.")
    return word_tokens


def bigram_counts():
    #Likehood on bigrams allows for next term prediction
    bgm = nltk.collocations.BigramAssocMeasures() #<-- likelihood a part

    finder = nltk.collocations.BigramCollocationFinder.from_words(load_entire_corpus_words())
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
        except:
            try:
                wcount[word]=1
            except:
                print ("COULD NOT COUNT WORD: "+str(word))
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

def loglikelyhood(fore_words,back_words,cutoff=10.83):
    print ("[debug] start loglikelyhood calc...")
    topic_signatures=[]

    fore_count_words,fore_word_length=word_counter(fore_words)
    back_count_words,back_word_length=word_counter(back_words)
    
    ratios={}
    c=0
    for word in set(fore_words):
        c+=1
        if not c%10000: print "AT "+str(c)+" / "+str(len(set(fore_words)))
        i_words=fore_count_words[word]
        i_size=fore_word_length
        
        #L1 <- dbinom(I.words, size = sum(I.words), prob = I.words/sum(I.words))
        l1=binom.pmf(i_words,i_size,i_words/i_size)

        #L2 <- dbinom(B.words, size = sum(B.words), prob = B.words/sum(B.words))
        b_words=back_count_words[word]
        b_size=back_word_length
        l2=binom.pmf(b_words,b_size,b_words/b_size)
    
        #L3 <- dbinom(I.words, size = sum(I.words), prob =  (I.words+B.words)/(sum(I.words)+sum(B.words)))
        prob=(i_words+b_words)/(i_size+b_size)
        l3=binom.pmf(i_words,i_size,prob)

        #L4 <- dbinom(B.words, size = sum(B.words), prob = (I.words+B.words)/(sum(I.words)+sum(B.words)) )
        prob=(i_words+b_words)/(i_size+b_size)
        l4=binom.pmf(b_words,b_size,prob)
        
        ratios[word]=2*(log(l1)+log(l2)-log(l3)-log(l4))
        
        if ratios[word]>cutoff:
            topic_signatures+=[word]

    return topic_signatures


def sample_calc_likelyhood():
    documents,sentences,sentences_topics=files2sentences(limit_topic='')
    foreground_text=documents[0].lower()

    foreground_words=nltk.wordpunct_tokenize(foreground_text)
    background_words=load_entire_corpus_words()
    
    topic_signatures=loglikelyhood(foreground_words,background_words)
    
    print ("GOT SIGNATURES: "+str(topic_signatures))

    return

def get_topic_topic_signatures(topic_id):
    #** potential speed-up by reusing background word list
    print ("[Start calculating topic signatures] for: "+str(topic_id)+"...")

    documents,sentences,sentences_topics=files2sentences(limit_topic='')
    
    foreground_words=[]
    background_words=[]

    #Include query sentence?
    for i,sentence in enumerate(sentences):
        if sentences_topics[i]==topic_id:
            foreground_words+=nltk.word_tokenize(sentence)
        background_words+=nltk.word_tokenize(sentence)
    
    print ("[done topic signatures] for: "+str(topic_id))
    return loglikelyhood(foreground_words, background_words)


if __name__=='__main__':
    branches=['bigram_counts']
    branches=['loglikelyhood']
    branches=['sample_calc_likelyhood']

    for b in branches:
        globals()[b]()


"""
other samples:  https://www.programcreek.com/python/example/107928/nltk.metrics.BigramAssocMeasures.likelihood_ratio

nltk.metrics.association.NgramAssocMeasures.likelihood_ratio

"""
























