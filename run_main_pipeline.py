from __future__ import division
import numpy as np
import random
from itertools import izip

from gensim import corpora
from gensim import models
from gensim import similarities

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import TEMP_DATA_PATH
from duc_reader import TOPIC_ID
from duc_reader import LIMIT_TOPICS
from duc_reader import SIM_MATRIX_PATH

from igraph import plot  #pycairo  #pycairo-1.17.1-cp27-cp27m-win_amd64.whl https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo
from performance import Performance_Tracker 

Perf=Performance_Tracker()



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_pipeline(verbose=True):
    global TOPIC_ID, LIMIT_TOPICS, SIM_MATRIX_PATH
    Perf.start()
    
    options=['print_sims']
    options=['print_entire_graph']

    #0/  Load query sentence
    vector_model='tfidf'

    query_sentence=get_query(TOPIC_ID)
    print ("Using query: "+str(query_sentence))

    #1/  LOAD
    #################################


    print ("1/  Loading sentences...")
    if LIMIT_TOPICS:
        if not TOPIC_ID:stop_bad=setup
        documents,sentences,sentences_topics=files2sentences(limit_topic=TOPIC_ID) #watch loading 2x data into mem
    else:
        documents,sentences,sentences_topics=files2sentences()
    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents.")
    print("---------------------------------")
    for i,sentence in enumerate(sentences):
        print ("Sample sentence.  Topic: "+str(sentences_topics[i])+": "+sentence)
        if i>2:break
        
    #Add query as V1
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,TOPIC_ID)


    #2/  Normalize corpus
    ##########################################

    ##print("---------------------------------")
    ##print("list of sentences:")
    ##print sentences
    ##print("---------------------------------")
    ##print("Tokenize sentences (After using PorterStemmer):")
    norm_sentences=tokenize_sentences(sentences)
    ##print norm_sentences
    ##print("---------------------------------")



    #STEP 3 : Index and vectorize
    #####################################################

    #We create a dictionary, an index of all unique values: <class 'gensim.corpora.dictionary.Dictionary'>
    #the Dictionary is used as an index to convert words into integers.
    dictionary = corpora.Dictionary(norm_sentences)
    ##print (dictionary)
    ##print("---------------------------------")
    ##print("Dictionary (token:id):")
    ##print(dictionary.token2id)
    ##print("---------------------------------")
    dictionary.save(TEMP_DATA_PATH+'doc_dict.dict') # store the dictionary, for future reference
    dictionary.save_as_text(TEMP_DATA_PATH+'doc_txt_dict.txt',sort_by_word=False) # SAVE the dictionary as a text file,
    #the format of doc_txt_dict.txt is: (id_1    word_1  document_frequency_1)

    #---------------------------------

    # compile corpus (vectors number of times each elements appears)
    #The "compile corpus" section actually converts each sentence into a list of integers ("integer" bag-of-words)
    #This raw_corpus is then fed into the tfidf model.
    raw_corpus = [dictionary.doc2bow(t) for t in norm_sentences]
    #Then convert tokenized documents to vectors: <type 'list'>
    print "Then convert tokenized documents to vectors: %s"% type(raw_corpus)
    #each document is a list of sentence (vectors) --> (id of the word, tf in this doc)
    ##print("raw_corpus:")
    ##print raw_corpus 
    #Save the vectorized corpus as a .mm file
    corpora.MmCorpus.serialize(TEMP_DATA_PATH+'doc_vectors.mm', raw_corpus) # store to disk
    print "Save the vectorized corpus as a .mm file"



    # STEP 4 : tfidf
    ###############################################
    corpus = corpora.MmCorpus(TEMP_DATA_PATH+'doc_vectors.mm')

    # Transform Text with TF-IDF
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    print "We initialize our TF-IDF transformation tool : %s"%type(tfidf)
   
    # corpus tf-idf
    corpus_tfidf = tfidf[corpus]
    print "We convert our vectors corpus to TF-IDF space : %s"%type(corpus_tfidf)
   


    # STEP 5 : Create similarity matrix of all files
    ###############################################
   
    index = similarities.MatrixSimilarity(tfidf[corpus])
    #print "We compute similarities from the TF-IDF corpus : %s"%type(index)
    index.save(TEMP_DATA_PATH+'sim_index.index')
    index = similarities.MatrixSimilarity.load(TEMP_DATA_PATH+'sim_index.index')
    
    sims = index[corpus_tfidf]
    #print "We get a similarity matrix for all sentences in the corpus %s"% type(sims)
    np.save(SIM_MATRIX_PATH,sims)



    # STEP 6:  Print sims 
    ###############################################
    if 'print_sims' in options:
        i=0
        for item in list(enumerate(sims)):
            i+=1
            if i>0:break
            sent_num1=item[0]
            for sent_num2,cosim_value in enumerate(item[1]):
                idx="("+str(sent_num1)+","+str(sent_num2)+")"
                cosim_str="%.9f" % cosim_value
                print ("AT: "+str(idx)+" sim: "+str(cosim_str))
                print ("  for sent1: "+str(sentences[sent_num1]))
                print ("   vs sent2: "+str(sentences[sent_num2]))
            
    print ("TOPIC ID: "+str(TOPIC_ID))
    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents.")
    print ("Done run_pipeline in: "+str(Perf.end())+"s")
    return
    #################################################


if __name__=='__main__':
    
    branches=['run_pipeline']

    for b in branches:
        globals()[b]()


