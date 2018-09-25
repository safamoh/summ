from __future__ import division
import numpy as np
import random
from itertools import izip
from duc_reader import *

from duc_reader import TOPIC_ID
from duc_reader import TEMP_DATA_PATH

from igraph import plot  #pycairo  #pycairo-1.17.1-cp27-cp27m-win_amd64.whl https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo
from performance import Performance_Tracker 

Perf=Performance_Tracker()

SIM_MATRIX_PATH=TEMP_DATA_PATH+"_"+TOPIC_ID+"_sim_matrix.npy"

#0vA# Sept 25, 2018  New config for TOPIC_ID
#                    > note:  random_walk_with_restart was removed

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


def load_sim_matrix_to_igraph():
    global TEMP_DATA_PATH,TOPIC_ID,LIMIT_TOPICS,SIM_MATRIX_PATH
    #LOAD STATE
    #####################################################

    query_sentence=get_query(TOPIC_ID)
    print ("Using query: "+str(query_sentence))

    if LIMIT_TOPICS:
        if not TOPIC_ID:stop_bad=setup
        documents,sentences,sentences_topics=files2sentences(limit_topic=TOPIC_ID) #watch loading 2x data into mem
    else:
        documents,sentences,sentences_topics=files2sentences()
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,TOPIC_ID)
    #
    #############################


    #Reload simulation matrix
    sims=np.load(SIM_MATRIX_PATH)
    

    #STEP A:  Zero node-to-node simularity diagonal to 0
    np.fill_diagonal(sims, 0)
    
    #STEP B:  Create iGraph
    G = igraph.Graph.Weighted_Adjacency(sims.tolist())
    G.vs['label'] = sentences   #node_names  # or a.index/a.columns

    return G,query_sentence

def fix_dendrogram(graph, cl):
    # takes a graph and an incomplete dendrogram 
    # https://lists.nongnu.org/archive/html/igraph-help/2014-02/msg00067.html
    already_merged = set()
    for merge in cl.merges:
        already_merged.update(merge)

    num_dendrogram_nodes = graph.vcount() + len(cl.merges)
    not_merged_yet = sorted(set(xrange(num_dendrogram_nodes)) - already_merged)
    if len(not_merged_yet) < 2:
        return

    v1, v2 = not_merged_yet[:2]
    cl._merges.append((v1, v2))
    del not_merged_yet[:2]

    missing_nodes = xrange(num_dendrogram_nodes,
            num_dendrogram_nodes + len(not_merged_yet))
    cl._merges.extend(izip(not_merged_yet, missing_nodes))
    cl._nmerges = graph.vcount()-1
    return cl

def filter_graph(g,the_type=''):
    #>>
    if the_type=='remove_low_weights':
        #/filter option by weight (but leads to non-complete graph
        #SELECT#  http://igraph.org/python/doc/igraph.EdgeSeq-class.html#select
        #g.es.select(weight_lt=0.20).delete()  #615k to 3.2k
        
        #1/
        b4edges=g.ecount()
        
        TRIM_WEIGHTS_LESS_THEN=0.13
        
        #SET CUT-OFF WEIGHT TO TRIM:
        
        #Past experiments:
            #g.es.select(weight_lt=0.20).delete()  #615k to 3.2 #33s
            #g.es.select(weight_lt=0.15).delete()  #615k to 8342 #772 seconds

        g.es.select(weight_lt=TRIM_WEIGHTS_LESS_THEN).delete()   #615k to ___

        after_edges=g.ecount()
        print ("Edges trimmed from "+str(b4edges)+" to: "+str(after_edges)+" reduce by: "+str(1-after_edges/b4edges)+"%")
        
    if False:
        # FILTER / TRIM GRAPH
        #/filter option by random
        sample_size=25
        subgraph_vertex_list = [v.index for v in random.sample(g.vs, sample_size)]
        subgraph = igraph.Graph.subgraph(g, subgraph_vertex_list)
        g=subgraph
        print ("FILTERING graph edge count to: "+str(g.ecount()))
    return g

def run_clustering_on_graph():
    global SIM_MATRIX_PATH
    method='fastgreedy'
    #method='betweenness'
    #method='walktrap'
    #method='spinglass'

    #STEP 1:  LOAD GRAPH ##########################
    #> check that graph exists
    if not os.path.exists(SIM_MATRIX_PATH):
        print (">> SIM MATRIX DOES NOT EXIST for: "+TOPIC_ID+".  Calling run_pipeline...")
        run_pipeline()
    g,query_sentence=load_sim_matrix_to_igraph()
    ###############################################
    #
    g=filter_graph(g)
    
    cluster_count=15
    
    print ("Running clustering ["+method+"] on graph...")
    Perf.start()
    if 'betweenness' in method:
        print ("**betweenness requires edge trimming.  Doing that now...")
        g=filter_graph(g,the_type='remove_low_weights')
        print ("Calculating edge betweenness...")
        communities=g.community_edge_betweenness(clusters=cluster_count,weights='weight') #directed=
        print ("Fixing/checking dendogram -- must be fully connected.")
        communities=fix_dendrogram(g, communities)
    #########################################################

    if 'fastgreedy' in method:
        #** only works with undirected graphs
        uG = g.as_undirected(combine_edges = 'mean') #Retain edge attributes: max, first.
        communities = uG.community_fastgreedy(weights = 'weight')
        
        #When an algorithm in igraph produces a VertexDendrogram, it may optionally produce a "hint" as well that tells us where to cut the dendrogram (i.e. after how many merges) to obtain a VertexClustering that is in some sense optimal. For instance, the VertexDendrogram produced by community_fastgreedy() proposes that the dendrogram should be cut at the point where the modularity is maximized. Running as_clustering() on a VertexDendrogram simply uses the hint produced by the clustering algorithm to flatten the dendrogram into a clustering, but you may override this by specifying the desired number of clusters as an argument to as_clustering().
        #As for the "distance" between two communities: it's a complicated thing because most community detection methods don't give you that information. They simply produce a sequence of merges from individual vertices up to a mega-community encapsulating everyone, and there is no "distance" information encoded in the dendrogram; in other words, the branches of the dendrogram have no "length". The best you can do is probably to go back to your graph and check the edge density between the communities; this could be a good indication of closeness. For example:
        

    #########################################################
    if 'walktrap' in method:
    #** only works with undirected graphs
        uG = g.as_undirected(combine_edges = 'mean') #Retain edge attributes: max, first.
        communities = uG.community_walktrap(weights = 'weight')


     #########################################################
    if 'spinglass' in method:
    #** only works with undirected graphs
        uG = g.as_undirected(combine_edges = 'mean') #Retain edge attributes: max, first.
        clusters    = uG.clusters()
        giant = clusters.giant()
        communities = giant.community_spinglass(weights = 'weight')

      
      

    time_clustering=Perf.end()
        

    clusters = communities.as_clustering(n=cluster_count) #Cut dendogram at level n. Returns VertexClustering object
                                                          #"When an algorithm in igraph produces a `VertexDendrogram`, it may optionally produce a "hint" as well that tells us where to cut the dendrogram 
    
    #Edges between clusters#  edges_between = g.es.select(_between=(comm1, comm2))
    
    output_clusters(g,communities,clusters)
    g.write_pickle(fname="save_clustered_graph.dat")

    print ("For topic: "+str(TOPIC_ID))
    print ("Done clustering took: "+str(time_clustering)+" seconds")
    
    if False:
        print ("Visualize clusters...")
        view_graph_clusters(g,clusters)
    return


def output_clusters(g,communities,clusters):
    #communities:  VertexDendogram 
    #clusters:  VertexClustering object -- http://igraph.org/python/doc/igraph.clustering.VertexClustering-class.html
    
    #Remove singletons
    #singletons = cg.vs.select(_degree = 0)
    #cg.delete_vertices(singletons)
    print_it=['samples','counts']
    print_it=['counts']
    
    #>  Print clusters with examples
    if 'samples' in print_it:
        i=-1
        for subgraph in clusters.subgraphs():
            i+=1
            print
            print ("Cluster #"+str(i)+" has node count: "+str(subgraph.vcount()))
        
            for idx, v in enumerate(subgraph.vs):
                print ("Node: "+str(v['label']))
                if idx>3:break

    #>  Print cluster counts
    if 'counts' in print_it:
        i=-1
        for subgraph in clusters.subgraphs():
            i+=1
            print("-----------------------------------")
            print ("Cluster #"+str(i)+" has node count: "+str(subgraph.vcount()))
            for idx, v in enumerate(subgraph.vs):
                print ("Node: "+str(v['label']))
                if idx>3:break
    
        print ("Total number of clusters: "+str(len(clusters)))
    return


def run_graph_on_sims():
    #STEP 1:  LOAD GRAPH ##########################
    G,query_sentence=load_sim_matrix_to_igraph()
    ###############################################
    #
    options=['print_entire_graph']
    options=[]
    
    
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
    print ("Total node count: "+str(len(G.vs)))
    print ("Total scores count: "+str(len(sorted_scores)))
    print ("Done for query: "+str(query_sentence))
    
    print ("Done run_graph_on_sims")
    return


def view_graph_clusters(g,clusters):
    
    #igraph selections:
    #vRes = G.vs.select(lambda x:x["name"]=="John" and x["age"]>20)
    
    #https://gist.github.com/rbnvrw/c2424fe3ff812da892a0
    
    #Change edge weights based on clusters
    
    # Set edge weights based on communities
    weights = {v: len(c) for c in clusters for v in c}
    g.es["weight"] = [weights[e.tuple[0]] + weights[e.tuple[1]] for e in g.es]

    visual_style = {}

    # Scale vertices based on degree
    outdegree = g.outdegree()
    visual_style["vertex_size"] = [x/max(outdegree)*25+50 for x in outdegree]
    
    # Set bbox and margin
    visual_style["bbox"] = (800,800)
    visual_style["margin"] = 100
    
    # Define colors used for outdegree visualization
    colours = ['#fecc5c', '#a31a1c']
    
    # Order vertices in bins based on outdegree
    bins = np.linspace(0, max(outdegree), len(colours))  
    digitized_degrees =  np.digitize(outdegree, bins)
    
    # Set colors according to bins
    g.vs["color"] = [colours[x-1] for x in digitized_degrees]
    
    # Also color the edges
    for ind, color in enumerate(g.vs["color"]):
            edges = g.es.select(_source=ind)
            edges["color"] = [color]
    # Don't curve the edges
    visual_style["edge_curved"] = False

    # Choose the layout
    N = g.vcount()

    visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=1000, area=N**3, repulserad=N**3)
            
    # Plot the graph
    #plot(g, "social_network.pdf", **visual_style)
    out=plot(g, **visual_style)
    out.save('igraph_visual.png')

    return


if __name__=='__main__':
    branches=['run_pipeline']
    branches=['run_clustering_on_graph']

    for b in branches:
        globals()[b]()



