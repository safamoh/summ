from __future__ import division
import os
import re
import numpy as np
import random
from itertools import izip
from collections import OrderedDict

from performance import Performance_Tracker 

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import TOPIC_ID
from duc_reader import LIMIT_TOPICS
from duc_reader import TEMP_DATA_PATH
from duc_reader import get_sim_matrix_path

from graph_utils import output_clusters
from graph_utils import view_graph_clusters
from graph_utils import load_sim_matrix_to_igraph
from graph_utils import fix_dendrogram
from graph_utils import filter_graph
from graph_utils import load_sim_matrix

from run_main_pipeline import run_pipeline

Perf=Performance_Tracker()


#0v1#  JC  Sept 30, 2018  Add cluster selection

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def run_random_walk_on_graph(topic_id):
    #STEP 1:  LOAD GRAPH ##########################
    #> check that graph exists
    if not os.path.exists(get_sim_matrix_path(topic_id)):
        print (">> SIM MATRIX DOES NOT EXIST for: "+TOPIC_ID+": "+str(get_sim_matrix_path(topic_id)))
        run_pipeline()
    g,query_sentence,sims=load_sim_matrix_to_igraph(local_topic_id=topic_id)
    query_node=g.vs.find(label=query_sentence)
    query_index=query_node.index
    ###############################################
    print ("/ calculating random walk with restart on graph size: "+str(g.vcount))
    return calc_random_walk_with_restart(g,query_index),query_index #sorted_scores


def run_clustering_on_graph():
    #method='fastgreedy'
    #method='betweenness'   #talking to much time, we will skip it
    #method='walktrap'
    #method='spinglass'  # 
    method='leading_eigenvector'


    #STEP 1:  LOAD GRAPH ##########################
    #> check that graph exists
    if not os.path.exists(get_sim_matrix_path(TOPIC_ID)):
        print (">> SIM MATRIX DOES NOT EXIST for: "+TOPIC_ID+": "+str(get_sim_matrix_path(TOPIC_ID)))
        run_pipeline()
    g,query_sentence,sims=load_sim_matrix_to_igraph()
    ###############################################
    #g=filter_graph(g)
    
    query_node=g.vs.find(label=query_sentence)
    query_index=query_node.index
    
    communities=[]
    clusters=[]
    print ("Running clustering ["+method+"] on graph...")
    Perf.start()
    if 'betweenness' in method:
        #cluster_count=15
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
        
        #When an algorithm in igraph produces a VertexDendrogram, it may optionally produce a "hint" as well that tells us where to cut the dendrogram (i.e. after how many merges)to obtain a VertexClustering that is in some sense optimal.
        #For instance, the VertexDendrogram produced by community_fastgreedy() proposes that the dendrogram should be cut at the point where the modularity is maximized.
        #Running as_clustering() on a VertexDendrogram simply uses the hint produced by the clustering algorithm to flatten the dendrogram into a clustering,
        #but you may override this by specifying the desired number of clusters as an argument to as_clustering().
        #As for the "distance" between two communities: it's a complicated thing because most community detection methods don't give you that information.
        #They simply produce a sequence of merges from individual vertices up to a mega-community encapsulating everyone, and there is no "distance" information encoded in the dendrogram;
        #in other words, the branches of the dendrogram have no "length". The best you can do is probably to go back to your graph and check the edge density between the communities; this could be a good indication of closeness. For example:
        #https://stackoverflow.com/questions/17413836/igraph-in-python-relation-between-a-vertexdendrogram-object-and-vertexclusterin
        

    #########################################################
    if 'walktrap' in method:
    #** only works with undirected graphs
        uG = g.as_undirected(combine_edges = 'mean') #Retain edge attributes: max, first.
        communities = uG.community_walktrap(weights = 'weight')


    #########################################################
    if 'spinglass' in method:
        #The implementation of the spinglass clustering algorithm that is included in igraph works on connected graphs only. 
        #You have to decompose your graph to its connected components, run the clustering on each of the connected components, 
        #and then merge the membership vectors of the clusterings manually.
        # clusters    = g.clusters()
        # giant       = clusters.giant() ## using the biggest component as an example, you can use the others here.
        # communities = giant.community_spinglass()
        #**graph must not be unconnected
        a=tbd
        u#G = g.as_undirected(combine_edges = 'mean') #Retain edge attributes: max, first.
        clustering =  uG.community_spinglass(weights = 'weight')

    #########################################################
    if 'leading_eigenvector' in method:
        #http://igraph.org/python/doc/igraph.Graph-class.html#community_leading_eigenvector
        uG = g.as_undirected(combine_edges = 'mean') #Retain edge attributes: max, first.
        clusters= uG.community_leading_eigenvector(clusters=None,weights = 'weight') #if clusters=None then tries as many as possible


    time_clustering=Perf.end()

    #Choose optimum number of communities
    if not clusters:
        num_communities = communities.optimal_count
        clusters = communities.as_clustering(n= num_communities) #Cut dendogram at level n. Returns VertexClustering object
                                                          #"When an algorithm in igraph produces a `VertexDendrogram`, it may optionally produce a "hint" as well that tells us where to cut the dendrogram 
                                                          
                                            
    # Calc weight of each cluster
    #########################################################
    #> weight = average cosine similarity between query and each node in cluster
    #> reuse sim calcs where possible
    #> note: subgraph index not same as g
    cluster_weights=[]
    for i,subgraph in enumerate(clusters.subgraphs()):
        edge_sums=0
        for idx, v in enumerate(subgraph.vs):
            #print ("GOT: "+str(v.attribute_names()))
            #print ("Node: "+str(v['label'])+" org id: "+str(v_idx))
            edge_sim=sims[query_index][v['s_idx']] #Use stored sentence index to look up old cosine sim value
            edge_sums+=edge_sim
        avg_weight=edge_sums/subgraph.vcount()
        cluster_weights+=[avg_weight]
        print ("Cluster #"+str(i)+" has node count: "+str(subgraph.vcount())+" avg weight: "+str(avg_weight))
                
    output_clusters(g,communities,clusters,cluster_weights=cluster_weights)
    g.write_pickle(fname="save_clustered_graph.dat")

    print ("For topic: "+str(TOPIC_ID))
    print ("Done clustering took: "+str(time_clustering)+" seconds")
    
    if False:
        print ("Visualize clusters...")
        view_graph_clusters(g,clusters)
        
    do_selection(g,clusters,cluster_weights,query_sentence)
    return


def calc_random_walk_with_restart(g,query_index):
    echo_top_scores=6
    if echo_top_scores:
        print ("---> Random walk top scores:")
    
    ## Calc random walk
    random_walk_with_restart=g.personalized_pagerank(reset_vertices=query_index)
    sorted_scores = sorted(zip(random_walk_with_restart, g.vs), key=lambda x: x[0],reverse=True) #G.vs is node id list

    ## Index by index (for O(n) look up of node)
    sorted_scores_idx=OrderedDict()
    rank=-1
    for score,vertex in sorted_scores:
        rank+=1
        sorted_scores_idx[vertex.index]=(score,vertex,rank)
        if rank<echo_top_scores:
            print ("RWS Rank #"+str(rank)+" is: "+str(score)+" sentence: "+vertex['label'])
    return sorted_scores_idx

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def do_selection(g,clusters,cluster_weights,query_sentence):
    ##
    #  Grab top sentences from each cluster
    ############################################

    weight_threshold=0.009   #; also consider a percentile value ie/ 90% of clusters

    #Grab query sentence info
    query_node=g.vs.find(label=query_sentence)
    query_index=query_node.index
    
    ##/  Random walk with restart calculation
    g_random_walk_scores=calc_random_walk_with_restart(g, query_index)
    
    ##1/  Sort clusters by weight
    #- rather then duplicating data structure (memory) adjust pointers to clusters
    ptr_tuple=[]
    for i,weight in enumerate(cluster_weights):
        ptr_tuple+=[(i,weight)]
    
    ptr_tuple=sorted(ptr_tuple,key=lambda x:x[1],reverse=True)

    ##2/  Filter clusters by weight threshold
    ptr_tuple_top=[]
    for i,weight in ptr_tuple:
        if weight>weight_threshold:
            ptr_tuple_top+=[(i,weight)]
    
    ##3/ Lookup random walk scores for clusters
    #i_cluster:  The index of the sub-graph (cluster)
    #vc_index:   The index of a vertex in the sub-graph
    #s_idx:      The original index of the sentence/vertex in g
    rws_lookup={}
    for i_cluster,weight in ptr_tuple_top:
        subgraph=clusters.subgraphs()[i_cluster]
        for vc_index, v in enumerate(subgraph.vs):
            s_idx=v['s_idx']
            walk_score,vertex,rank=g_random_walk_scores[s_idx]
            #Store lookup between vertex index and walk scores
            if not i_cluster in rws_lookup: rws_lookup[i_cluster]=[]
            rws_lookup[i_cluster]+=[(vc_index,s_idx,walk_score)]
    
    ##4/  Random walk sort within each cluster
    print ()
    for i_cluster,weight in ptr_tuple_top:
        print ("---- Selection on cluster #"+str(i_cluster)+"------ ")
        subgraph=clusters.subgraphs()[i_cluster]
        
        #Sort walk scores for each cluster
        rws_sorted=sorted(rws_lookup[i_cluster], key=lambda x:x[2],reverse=True) #Sort by walk score
        
        ##/  Select top n  OR  % proportional
        top_n=3
        proportional_size=subgraph.vcount()/g.vcount()

        c=0
        for vc_index,s_idx,walk_score in rws_sorted:
            #Ignore original query
            if s_idx==query_index:continue  #Skip query_index
            c+=1
            vertex=subgraph.vs[vc_index]
            print ("Cluster #"+str(i_cluster)+" Top Score #"+str(c)+": %.9f"%walk_score+" >"+str(vertex['label']))
            if c==top_n:break
        
    print ("Done do_selection")   
    return

def select_top_cos_sims(topic_id='',top_n=10,verbose=True):
    #Recall, sentences variable does not have query sentence
    #Recall, sim matrix has query sentence values at 0

    top_sentences=[]

    print ("SELECT TOP COS SIM FOR TOPIC: "+str(topic_id)+"--------")
    #STEP 1:  LOAD COS SIM MATRIX #################
    query_sentence=get_query(topic_id)
    print ("Query sentence: "+str(query_sentence))

    ## Get sentences
    documents,sentences,sentences_topics=files2sentences(limit_topic=topic_id) #watch loading 2x data into mem
    
    ## Load sim matrix
    try:
        sims=load_sim_matrix(topic_id,zero_node2node=False) #Don't zero out diagonal to test scoring sort
    except:
        print ("NO SIM MATRIX FOR: "+topic_id)
        print ("Auto run pipeline to calc sim matrix")
        run_pipeline(use_specific_topic=topic_id)
        sims=load_sim_matrix(topic_id,zero_node2node=False) #Don't zero out diagonal to test scoring sort

    query_sims=sims.tolist()[0] #First is query -- but set to 0 for itself.
    
    ## Sort cos sims
    sorted_scores = sorted(zip(query_sims, range(len(query_sims))), key=lambda x: x[0],reverse=True) #G.vs is node id list
    
    ## Validate top score is index 1
    if not sorted_scores[0][1]==0:
        print ("Error on sim matrix load")
        hard_stop=expect_0_index
    ## Validate that scores include query string
    if not len(query_sims)==(len(sentences)+1):
        print ("Scores does not include query index at 0")
        hard_stop=expect_lengths
        
    sentences.insert(0,query_sentence)
    ## output scores
    c=-1
    for cos_sim,idx in sorted_scores:
        c+=1
        sentence=sentences[idx]
        if c>0:
            if verbose:
                print ("#"+str(c)+" score: "+str(cos_sim)+" sentence idx: "+str(idx)+" sentence: "+str(sentence))
            top_sentences+=[sentence]
        if c==top_n: break
    return top_sentences


if __name__=='__main__':
    branches=['run_clustering_on_graph']
    branches=['select_top_cos_sims']

    for b in branches:
        globals()[b]()





















