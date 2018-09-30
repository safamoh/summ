from __future__ import division
import os
import re
import numpy as np
import random
from itertools import izip

from performance import Performance_Tracker 

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import TOPIC_ID
from duc_reader import LIMIT_TOPICS
from duc_reader import TEMP_DATA_PATH
from duc_reader import SIM_MATRIX_PATH

from graph_utils import output_clusters
from graph_utils import view_graph_clusters
from graph_utils import load_sim_matrix_to_igraph
from graph_utils import fix_dendrogram
from graph_utils import filter_graph

from run_main_pipeline import run_pipeline

Perf=Performance_Tracker()


#0v1#  JC  Sept 30, 2018  Add cluster selection

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_clustering_on_graph():
    global SIM_MATRIX_PATH
    #method='fastgreedy'
    #method='betweenness'   #talking to much time, we will skip it
    #method='walktrap'
    #method='spinglass'  # 
    method='leading_eigenvector'


    #STEP 1:  LOAD GRAPH ##########################
    #> check that graph exists
    if not os.path.exists(SIM_MATRIX_PATH):
        print (">> SIM MATRIX DOES NOT EXIST for: "+TOPIC_ID+": "+str(SIM_MATRIX_PATH))
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


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def do_selection(g,clusters,cluster_weights,query_sentence):
    ##
    #  Grab top sentences from each cluster
    ############################################

    weight_threshold=0.009   #; also consider a percentile value ie/ 90% of clusters

    #Grab query sentence info
    query_node=g.vs.find(label=query_sentence)
    query_index=query_node.index
    
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
    
    ##3/  Random walk sort within each cluster
    print ()
    for i,weight in ptr_tuple_top:
        print ("---- Selection on cluster #"+str(i)+"------ ")

        subgraph=clusters.subgraphs()[i]
        subgraph.add_vertex(query_node)
        sub_query_index=subgraph.vcount()-1
        #v1 = g.vs[g.vcount()-1]
        
        ##/  Random walk with restart on cluster
        sub_random_walk_with_restart=subgraph.personalized_pagerank(reset_vertices=sub_query_index)
        
        ##/  Sort
        sub_sorted_scores = sorted(zip(sub_random_walk_with_restart, subgraph.vs), key=lambda x: x[0],reverse=True)
        
        ##/  Select top n  OR  % proportional
        top_n=3
        proportional_size=subgraph.vcount()/g.vcount()
        
        c=0
        for score,vertex in sub_sorted_scores[1:top_n+1]: #First is query_node
            #Check that not matching query node
            c+=1
            print ("Cluster #"+str(i)+" Top Score #"+str(c)+": %.9f"%score+" >"+str(vertex['label']))
            
    print ("Done do_selection")   
    return


if __name__=='__main__':
    branches=['run_clustering_on_graph']

    for b in branches:
        globals()[b]()





















