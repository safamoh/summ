from read import *


def run_pipeline(verbose=True):
    global TOPIC_ID, LIMIT_TOPICS
    
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
    np.save(TEMP_DATA_PATH+"sim_matrix.npy",sims)



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
            
    return
    #################################################


def run_graph_on_sims():
    global TEMP_DATA_PATH,TOPIC_ID,LIMIT_TOPICS
    
    options=['print_entire_graph']
    options=[]
    
    #LOAD STATE
    #####################################################
    print ("[]TODO: consider keeping in same pipeline as above -- otherwise, watch training parameters are same")

    topic_id='d301i' #
    query_sentence=get_query(topic_id)
    print ("Using query: "+str(query_sentence))

    if LIMIT_TOPICS:
        if not TOPIC_ID:stop_bad=setup
        documents,sentences,sentences_topics=files2sentences(limit_topic=TOPIC_ID) #watch loading 2x data into mem
    else:
        documents,sentences,sentences_topics=files2sentences()
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,topic_id)
    #
    #############################


    #Reload simulation matrix
    sims=np.load(TEMP_DATA_PATH+"sim_matrix.npy")
    

    #STEP A:  Zero node-to-node simularity diagonal to 0
    np.fill_diagonal(sims, 0)
    
    #STEP B:  Create iGraph
    G = igraph.Graph.Weighted_Adjacency(sims.tolist())
    G.vs['label'] = sentences   #node_names  # or a.index/a.columns

    #STEP C:  Query index or vector (alternatively matrix 1 at query, zero otherwise)
    query_node_id=0
    print ("Query from node: "+str(G.vs[query_node_id]))
    
    #STEP D:  Random walk with restart
    random_walk_with_restart=G.personalized_pagerank(reset_vertices=query_node_id)
    print ("GOT RANDOM walk scores: "+str(random_walk_with_restart[:5])+"...")
    #  Options:
    #         - weights - edge weights to be used. Can be a sequence or iterable or even an edge attribute name.
    #         - returns list
    
    #STEP E:  Sort and chose top scores
    sorted_scores = sorted(zip(random_walk_with_restart, G.vs), key=lambda x: x[0],reverse=True) #G.vs is node id list
    
    #STEP F:  Output top sentences
    #         - top match is query node itself
    c=0
    for score,vertex in sorted_scores:
        c+=1
        if c>6:break
        if c>1:
            print ("Top Score: %.9f"%score+" >"+vertex['label'])

    if 'print_entire_graph' in options:
        c=0
        for edge in G.es:
            c+=1
            if c>10:break

            source_vertex_id = edge.source
            target_vertex_id = edge.target
            source_vertex = G.vs[source_vertex_id]
            target_vertex = G.vs[target_vertex_id]
            
            print ("From: "+str(source_vertex)+" to: "+str(target_vertex))
            print ("Weight: "+str(edge['weight']))
            if False and edge['weight']>0.15:
                print ("------------------")
                print ("From: "+str(source_vertex)+" to: "+str(target_vertex))
                print ("Weight: "+str(edge['weight']))
            
    print
    print ("Done for query: "+str(query_sentence))
    
    print ("Done run_graph_on_sims")
    return

if __name__=='__main__':
    branches=['run_pipeline']
    branches+=['run_graph_on_sims']

#    branches=['run_graph_on_sims']
    for b in branches:
        globals()[b]()



















