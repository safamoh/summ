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
from duc_reader import TOPIC_ID
from duc_reader import LIMIT_TOPICS
from duc_reader import get_sim_matrix_path

from igraph import plot  #pycairo  #pycairo-1.17.1-cp27-cp27m-win_amd64.whl https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo
from performance import Performance_Tracker 

Perf=Performance_Tracker()



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_pipeline(verbose=True,use_all_topics=False,use_specific_topic=''):
    global TOPIC_ID, LIMIT_TOPICS
    Perf.start()
    if use_specific_topic:
        local_topic_id=use_specific_topic
    else:
        local_topic_id=TOPIC_ID
    
    options=['print_entire_graph']
    options=[]
    options=['print_sims']

    #0/  Load query sentence
    vector_model='tfidf'

    query_sentence=get_query(local_topic_id)
    print ("Using query: "+str(query_sentence))

    #1/  LOAD
    #################################


    print ("1/  Loading sentences...")
    if not LIMIT_TOPICS or use_all_topics:
        requires_selection_of=sentence_properly
        documents,sentences,sentences_topics=files2sentences(limit_topic='')
    else:
        documents,sentences,sentences_topics=files2sentences(limit_topic=local_topic_id)

    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents. "+str(len(set(sentences_topics)))+" topics.")
    print("---------------------------------")
    for i,sentence in enumerate(sentences):
        #print ("Sample sentence.  Topic: "+str(sentences_topics[i])+": "+sentence)
        if i>2:break
        
    #Add query as V1
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,local_topic_id)

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
    np.save(get_sim_matrix_path(local_topic_id),sims)


    # STEP 6:  Print sims 
    ###############################################
    if 'print_sims' in options:
        i=0
        for item in list(enumerate(sims)):
            i+=1
            #            if i>0:break
            sent_num1=item[0]
            for sent_num2,cosim_value in enumerate(item[1]):
                idx="("+str(sent_num1)+","+str(sent_num2)+")"
                cosim_str="%.9f" % cosim_value
                if False:
                    print ("AT: "+str(idx)+" sim: "+str(cosim_str))
                    print ("  for sent1: "+str(sentences[sent_num1]))
                    print ("   vs sent2: "+str(sentences[sent_num2]))

            

    print ("TOPIC ID: "+str(local_topic_id))
    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents.")
    print ("Done run_pipeline in: "+str(Perf.end())+"s")

    # STEP 7:  Vectorize sentences
    #TBD
    if False:
        print ("SENTENCE 0: "+str(sentences[0]))
        print ("Has normalize bow: "+str(raw_corpus[0]))
        sent_vector0 = tfidf[raw_corpus[0]]
        sent_vector1 = tfidf[raw_corpus[1]]
        print ("Sentence 0 verctor: "+str(sent_vector0))
        print ("Sentence 0 verctor type: "+str(type(sent_vector0)))
        
        featureVec=sent_vector0
        print ("Normalize vector before average?")
        num_features=len(sent_vector0)
        
        vectors=[]
        vectors+=[matutils.unitvec(sent_vector0)]
        vectors+=[matutils.unitvec(sent_vector0)]
        vectors+=[matutils.unitvec(sent_vector1)]
        print ("UNIT: "+str(vectors))
        
        if False:
            #num_features=sent_vector0[0][0].shape  #     vector.shape (100,)
            print ("FEATURES: "+str(num_features))
            sumVec = np.zeros((num_features,), dtype="float32")
            sumVec = np.add(featureVec)
        
            avgVec = np.divide(sumVec, 1)
        else:
    #        mean = matutils.unitvec(np.array(vectors).mean(axis=0)).astype(REAL)
            mean = np.array(vectors).mean(axis=0)
            print ("GOT MEAN: "+str(mean))
    #        mean = matutils.unitvec(np.array(vectors).mean(axis=0))
    #        dists = dot(vectors, mean)
    
    #If you multiply the BoW vector with the word embedding matrix and divide by the total number of words in the document then you have the average word2vec representation. This contains mostly the same information as BoW but in a lower dimensional encoding. You can actually train a model to recover which words were used in the document from the average word2vec vector. So, you aren't losing very much information by compressing the representation like this.
    
    #def intra_inter(model, test_docs, num_pairs=10000):
    #    # split each test document into two halves and compute topics for each half
    #    part1 = [model[id2word_wiki.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    #    part2 = [model[id2word_wiki.doc2bow(tokens[len(tokens) / 2 :])] for tokens in test_docs]
    #    
    #    # print computed similarities (uses cossim)
    #    print("average cosine similarity between corresponding parts (higher is better):")
    #    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))
    #
    #    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    #    print("average cosine similarity between 10,000 random parts (lower is better):")    
    #    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))
        
    #    print ("Average: ____: "+str(avgVec))
        

    return


if __name__=='__main__':
    
    branches=['run_pipeline']

    for b in branches:
        globals()[b]()











