from __future__ import division
import numpy as np
import random
from itertools import izip

from gensim import corpora
from gensim import models
from gensim import similarities
from gensim import utils, matutils

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import TEMP_DATA_PATH
from duc_reader import get_sim_matrix_path
#from duc_reader import TOPIC_ID
#from duc_reader import LIMIT_TOPICS
from duc_reader import get_list_of_all_topics

from igraph import plot  #pycairo  #pycairo-1.17.1-cp27-cp27m-win_amd64.whl https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo
from performance import Performance_Tracker 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

from run_main_pipeline_sklearn import cosine_sim_algs
from run_main_pipeline_sklearn import print_sims

Perf=Performance_Tracker()

def run_pipeline(verbose=True,create_all_topics_vectorizer=False,use_all_topics_vectorizer=False,local_topic_id=''):
    global USE_SKLEARN_MODELS
    
    #STEP 1:  Build vectorizer
    #STEP 2:  Do sim matrix
    Perf.start()

    #0/  Load query sentence
    vector_model='tfidf'
    tfidf_filename=TEMP_DATA_PATH+'tfidf_model_'+local_topic_id+'.mm'
    
    print ("########################################")
    print ("#")
    if create_all_topics_vectorizer:
        print ("# Create vectorizer using all topics/sentences")
        print ("# - saved as: "+tfidf_filename)
        print ("# - run_pipeline called twice. First time builds it.  Second time tokenizes topic sentences for sim matrix")
    if use_all_topics_vectorizer:
        print ("# Use the all_topics vectorizer")
        print ("# - assume this is second run.  Load topic sentences and tokenize them using vector")
        print ("# - Then create sim matrix")


    #1/  LOAD
    #################################
    print ("1/  Loading sentences...")
    if create_all_topics_vectorizer:
        documents,sentences,sentences_topics=files2sentences(limit_topic='')
        #Add all query sentences
        for topic_id in get_list_of_all_topics():
            query_sentence=get_query(topic_id)
            sentences.insert(0,query_sentence)
            sentences_topics.insert(0,topic_id)
        print ("Done building sentences...")
    else:
        documents,sentences,sentences_topics=files2sentences(limit_topic=local_topic_id)

        #Add query as V1
        query_sentence=get_query(local_topic_id)
        print ("Using query: "+str(query_sentence))
        sentences.insert(0,query_sentence)
        sentences_topics.insert(0,local_topic_id)

    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents. "+str(len(set(sentences_topics)))+" topics.")
    print("---------------------------------")
    for i,sentence in enumerate(sentences):
        print ("Sample sentence.  Topic: "+str(sentences_topics[i])+": "+sentence)
        if i>2:break
        
    
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
    dictionary_filename=TEMP_DATA_PATH+'doc_dict'+local_topic_id+'.dict'
    dictionary_filename_txt=TEMP_DATA_PATH+'doc_dict'+local_topic_id+'.txt'

    #We create a dictionary, an index of all unique values: <class 'gensim.corpora.dictionary.Dictionary'>
    #the Dictionary is used as an index to convert words into integers.
    dictionary = corpora.Dictionary(norm_sentences)
    ##print (dictionary)
    ##print("---------------------------------")
    ##print("Dictionary (token:id):")
    ##print(dictionary.token2id)
    ##print("---------------------------------")
    dictionary.save(dictionary_filename) # store the dictionary, for future reference
    dictionary.save_as_text(dictionary_filename_txt,sort_by_word=False) # SAVE the dictionary as a text file,
    #the format of doc_txt_dict.txt is: (id_1    word_1  document_frequency_1)

    #---------------------------------

    # compile corpus (vectors number of times each elements appears)
    #The "compile corpus" section actually converts each sentence into a list of integers ("integer" bag-of-words)
    #This raw_corpus is then fed into the tfidf model.
    raw_corpus = [dictionary.doc2bow(t) for t in norm_sentences]

    #Save the bow corpus as a .mm file
    corpora.MmCorpus.serialize(TEMP_DATA_PATH+'doc_vectors.mm', raw_corpus) # store to disk
    print "Save the vectorized corpus as a .mm file"
    
    # STEP 4 : tfidf
    ###############################################
    corpus = corpora.MmCorpus(TEMP_DATA_PATH+'doc_vectors.mm')
    

    if use_all_topics_vectorizer: 
        #LOAD GLOBAL MODEL
        tfidf_filename=TEMP_DATA_PATH+'tfidf_model_'+'.mm'   #no topic id
        print ("Use all_topics vectorizing model: "+str(tfidf_filename))
        tfidf=models.TfidfModel.load(tfidf_filename)
    else:
        #SAVE TOPIC MODEL
        # Transform Text with TF-IDF
        tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        tfidf.save(tfidf_filename)
        
    if create_all_topics_vectorizer: 
        print ("If created, then assume used on next call...")
    else:
        print ("Use tfidf model: "+str(tfidf_filename))
       
        # corpus tf-idf
        corpus_tfidf = tfidf[corpus]
        print "[debug1] We convert our vectors corpus to TF-IDF space : %s"%type(corpus_tfidf)
    
        # STEP 5 : Create similarity matrix of all files
        ###############################################
       
        index = similarities.MatrixSimilarity(tfidf[corpus])
        #print "We compute similarities from the TF-IDF corpus : %s"%type(index)
        index.save(TEMP_DATA_PATH+'sim_index.index')
        index = similarities.MatrixSimilarity.load(TEMP_DATA_PATH+'sim_index.index')
        
        sims = index[corpus_tfidf]
        #print "We get a similarity matrix for all sentences in the corpus %s"% type(sims)
        np.save(get_sim_matrix_path(local_topic_id),sims)
    
    
        print_sims(sims,sentences)
    return


if __name__=='__main__':
    
    branches=['run_pipeline']

    for b in branches:
        globals()[b]()











