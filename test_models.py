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

Perf=Performance_Tracker()


DEV_MAX_DOCS=1000


def run_pipeline(verbose=True,create_all_topics_vectorizer=False,use_all_topics_vectorizer=False,local_topic_id=''):
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
        
    sentences=sentences[:DEV_MAX_DOCS]

    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents. "+str(len(set(sentences_topics)))+" topics.")
    print("---------------------------------")
    for i,sentence in enumerate(sentences):
        print ("Sample sentence.  Topic: "+str(sentences_topics[i])+": "+sentence)
        if i>2:break
        

#    if create_all_topics_vectorizer or not use_all_topics_vectorizer: #Create specific vectorizer
    print ("Creating vectorizer... using "+str(len(sentences))+" sentences")
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
    #The "compile corpus" section actually converts each sentence into a 
    #list of integers ("integer" bag-of-words)
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
    

    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    tfidf.save(tfidf_filename)
        
        
    if create_all_topics_vectorizer: 
        print ("If created, then assume used on next call...")
    else:
        print ("Use tfidf model: "+str(tfidf_filename))
       
        # corpus tf-idf
        corpus_tfidf = tfidf[corpus]
        print "We convert our vectors corpus to TF-IDF space : %s"%type(corpus_tfidf)
    
        # STEP 5 : Create similarity matrix of all files
        ###############################################
       
#        index = similarities.MatrixSimilarity(tfidf[corpus])
        index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=len(dictionary))

#        print ("TYPE model: "+str(tfidf))
#        print ("TYPE corpus: "+str(corpus))
#        print ("TYPE index: "+str(index))

        #TYPE model: TfidfModel(num_docs=1000, num_nnz=10691)
        #TYPE corpus: TfidfModel(num_docs=1000, num_nnz=10691)
        #TYPE corpus: MmCorpus(1000 documents, 3268 features, 10691 non-zero entries)
        #TYPE index: MatrixSimilarity<1000 docs, 3268 features>
        
        #TYPE model: TfidfModel(num_docs=1000, num_nnz=10691)
        #TYPE corpus: MmCorpus(1000 documents, 3268 features, 10691 non-zero entries)
        #TYPE index: <gensim.similarities.docsim.SparseMatrixSimilarity object at 0x000000002A146EF0>
        
        
#         pipeline = Pipeline([
#    ("vect", CountVectorizer(min_df=0, stop_words="english")),
#    ("tfidf", TfidfTransformer(use_idf=False))])
#  tdMatrix = pipeline.fit_transform(docs, cats)

    
        if True: #Jan 19
            
#            documents = (
#                "The sky is blue",
#                "The sun is bright",
#                "The sun in the sky is bright",
#                "We can see the shining sun, the bright sun"
#                )
#

            print ("corpus type: "+str(type(documents))) #TUPLE
            print ("FIRST DOCUMENT TYPE: "+str(type(documents[0])))
            print ("FIRST DOCUMENT: "+str(documents[0]))
            
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents) #Fit_transform=fit, transform
            
            print tfidf_matrix.shape
            
            if False: #sentences
                #OPTIONS/  (sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english', vocabulary=vocabulary)
                tfidf_sentences=tfidf_vectorizer.transform(sentences)
                
            if True: #retokenize sentences
                #norm_sentences:  word tokenized
                sentences_stemmed=[]
                for sentence_words_list in norm_sentences:
                    sentence=" ".join(sentence_words_list)
                    sentences_stemmed+=[sentence]
            print ("FIRST S: "+str(sentences_stemmed[0]))
            
            tfidf_sentences=tfidf_vectorizer.transform(sentences_stemmed)
            
            #Now we have the TF-IDF matrix (tfidf_matrix) for each document

#            #we can calculate the Cosine Similarity between the first document The sky is blue with each of the other documents of the set
#            from sklearn.metrics.pairwise import cosine_similarity
#            cs=cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
#            print ("YO: "+str(cs))
#            print ("Do full matrix")
#            cs=cosine_similarity(tfidf_matrix, tfidf_matrix)
#            print ("Done")
#            
#            #In case others were wondering like I did, in this case linear_kernel is equivalent to cosine_similarity because the TfidfVectorizer produces normalized vectors
#            
#            from sklearn.metrics.pairwise import linear_kernel
#            #batch if memory issue https://stackoverflow.com/questions/46435220/calculating-similarity-between-tfidf-matrix-and-predicted-vector-causes-memory-o
#            cs2 = linear_kernel(tfidf_matrix, tfidf_matrix).flatten()
#            
#            print ("TYPE1: "+str(type(cs)))
#            print ("TYPE2: "+str(type(cs2))) #ndarray
            
            sims1=cosine_similarity(tfidf_sentences,tfidf_sentences)

#            print ("TYPE1: "+str(type(sims1)))
        
#            sims1 = index[corpus_tfidf]
#        #print "We get a similarity matrix for all sentences in the corpus %s"% type(sims1)
#        np.save(get_sim_matrix_path(local_topic_id),sims1)
    
    
    
            # STEP 6:  Print sims1 
            ###############################################
            options=[]
            options=['print_sims1']
        
            if 'print_sims1' in options:
                i=0
                j=0
                for item in list(enumerate(sims1)):
                    i+=1
                    #            if i>0:break
                    sent_num1=item[0]
                    sent_text1=sentences[sent_num1].strip()
                    for sent_num2,cosim_value in enumerate(item[1]):
                        sent_text2=sentences[sent_num2].strip()
                        if not sent_text1 or not sent_text2:continue
                        j+=1
                        idx="("+str(sent_num1)+","+str(sent_num2)+")"
                        cosim_str="%.9f" % cosim_value
                        if True and j<3:
                            print ("AT: "+str(idx)+" sim: "+str(cosim_str))
                            print ("  for sent1: >"+str(sentences[sent_num1])+"<")
                            print ("   vs sent2: >"+str(sentences[sent_num2])+"<")
                        



        if False:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import linear_kernel
            from sklearn.metrics import pairwise_distances
            from sklearn.metrics.pairwise import cosine_similarity 
    
            norm_sentences
            tfidf = TfidfVectorizer()
            tfidf_cluster = tfidf.fit_transform(norm_sentences)
            # Tranform the corpus using the trained tfidf
    #        tfidf_corpus = tfidf.transform(norm_sentences)
            X = pairwise_distances(tfidf_cluster)


        
        
        

        if False:
      
            
            print ("----> sklearn")
            from gensim.sklearn_api import TfIdfTransformer
            # Transform the word counts inversely to their global frequency using the sklearn interface.
            model = TfIdfTransformer(dictionary=dictionary)
            
            #1/
            # Transform the word counts inversely to their global frequency using the sklearn interface.
            # returns sparse-prepresentation of document term matrix (doc term matrix representation of training set)
            tfidf_corpus = model.fit_transform(raw_corpus)
#            print ("SHAPE: "+str(tfidf_corpus.shape))
    
            
            #2/ Transform test set
            new_tfidf = model.transform(raw_corpus)
            
            # returns a sparse-representation of a document-term matrix. It is the document-term matrix representation of your training set. You would then need to transform the testing set with the same model
    
            
            print ("GOT MODEL: "+str(type(model)))
    #        print ("GOT corpus: "+str(type(tfidf_matrix))) #List? 
    #        print ("FIRST: "+str(tfidf_matrix[0]))
            print ("FIRST: "+str(new_tfidf[0]))
            
            from sklearn.metrics.pairwise import linear_kernel
            from sklearn.metrics import pairwise_distances
            from sklearn.metrics.pairwise import cosine_similarity 
            
            tfidf_cluster=new_tfidf
            # Cosine similarity
            cos_similarity = np.dot(new_tfidf, tfidf_cluster.T).A
            avg_similarity = np.mean(cos_similarity, axis=1)
            
    #?        cosine = cosine_similarity(new_tfidf,new_tfidf)
    #        cosine = cosine_similarity(tfidf_matrix[length-1], tfidf_matrix)
    #        cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
            print (cosine)
    
    #distance_matrix = pairwise_distances(query_vector, 
    #                                     svd_matrix, 
    #                                     metric='cosine', 
    #                                     n_jobs=-1)
    
    
    
    
#        X = pairwise_distances(new_tfidf)#, metric = metrics,n_jobs = -2 )
#        X = pairwise_distances(tfidf_matrix)#, metric = metrics,n_jobs = -2 )
        
        
#        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        #cosine_similarities = linear_kernel(new_tfidf, new_tfidf)
        #print ("CS: "+str(cosine_similarities[0]))
        
        def find_similar(tfidf_matrix, index, top_n = 5):
            cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
            related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
            return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

        #GOT MODEL: <class 'gensim.sklearn_api.tfidf.TfIdfTransformer'>
        #GOT corpus: <type 'list'>

        
        
        
        #print "We compute similarities from the TF-IDF corpus : %s"%type(index)
        index.save(TEMP_DATA_PATH+'sim_index.index')
        index = similarities.MatrixSimilarity.load(TEMP_DATA_PATH+'sim_index.index')
        
        sims = index[corpus_tfidf]
        #print "We get a similarity matrix for all sentences in the corpus %s"% type(sims)
        np.save(get_sim_matrix_path(local_topic_id),sims)
    
    
    
        # STEP 6:  Print sims 
        ###############################################
        options=[]
        options=['print_sims']
    
        if 'print_sims' in options:
            i=0
            j=0
            for item in list(enumerate(sims)):
                i+=1
                #            if i>0:break
                sent_num1=item[0]
                sent_text1=sentences[sent_num1].strip()
                for sent_num2,cosim_value in enumerate(item[1]):
                    sent_text2=sentences[sent_num2].strip()
                    if not sent_text1 or not sent_text2:continue
                    j+=1
                    idx="("+str(sent_num1)+","+str(sent_num2)+")"
                    cosim_str="%.9f" % cosim_value
                    if True and j<3:
                        print ("AT: "+str(idx)+" sim: "+str(cosim_str))
                        print ("  for sent1: >"+str(sentences[sent_num1])+"<")
                        print ("   vs sent2: >"+str(sentences[sent_num2])+"<")
    
        print ("TOPIC ID: "+str(local_topic_id))
        print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents.")
        print ("Done run_pipeline in: "+str(Perf.end())+"s")

    return


if __name__=='__main__':
    
    branches=['run_pipeline']

    for b in branches:
        globals()[b]()











