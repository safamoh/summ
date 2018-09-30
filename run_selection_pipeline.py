from __future__ import division
import numpy as np
import random
from itertools import izip

from performance import Performance_Tracker 

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import TOPIC_ID
from duc_reader import TEMP_DATA_PATH
from duc_reader import SIM_MATRIX_PATH

from graph_utils import output_clusters
from graph_utils import view_graph_clusters
from graph_utils import load_sim_matrix_to_igraph
from graph_utils import fix_dendrogram
from graph_utils import filter_graph

from run_main_pipeline import run_pipeline

Perf=Performance_Tracker()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_graph_on_sims():
    #STEP 1:  LOAD GRAPH ##########################
    G,query_sentence,sims=load_sim_matrix_to_igraph()
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


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


if __name__=='__main__':
    branches=['run_clustering_on_graph']

    for b in branches:
        globals()[b]()



