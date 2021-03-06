from __future__ import division
import sys
import os
import re
import numpy as np
#import scipy.stats as ss
import random
import operator
from itertools import izip
from collections import OrderedDict

from performance import Performance_Tracker 

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import LIMIT_TOPICS
from duc_reader import TEMP_DATA_PATH
from duc_reader import get_sim_matrix_path

from graph_utils import output_clusters
from graph_utils import view_graph_clusters
from graph_utils import load_sim_matrix_to_igraph
from graph_utils import fix_dendrogram
from graph_utils import filter_graph
from graph_utils import load_sim_matrix
from graph_utils import add_vertex_to_graph
from graph_utils import calc_rank_percent_distribution
from graph_utils import normalize_max_min

from graph_utils import iter_graph_edges

from run_main_pipeline import run_pipeline
from gensim import corpora
from gensim import models
from gensim import similarities
from scipy.sparse import csr_matrix

import markov_clustering as mc

from igraph.clustering import VertexClustering
from duc_reader import DOCS_SOURCE

from vis_graph import _plot


Perf=Performance_Tracker()

#0v9# Jan 22, 2019  Max acceptable sentence length: 90 --> d0602b 2006
#0v8# Jan 21, 2019  Require min 7 words for summary

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#igraph to scipy sparse matrix
#https://github.com/igraph/python-igraph/issues/72


def to_sparse(graph, weight_attr=None):
    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)
    return csr_matrix((weights, zip(*edges)))


def run_random_walk_on_graph(topic_id):
    #STEP 1:  LOAD GRAPH ##########################
    #> check that graph exists
    if not os.path.exists(get_sim_matrix_path(topic_id)):
        print (">> SIM MATRIX DOES NOT EXIST for: "+topic_id+": "+str(get_sim_matrix_path(topic_id)))
        run_pipeline()
    g,query_sentence,sims=load_sim_matrix_to_igraph(local_topic_id=topic_id)
    query_node=g.vs.find(label=query_sentence)
    query_index=query_node.index
    ###############################################
    print ("/ calculating random walk with restart on graph size: "+str(g.vcount))
    return calc_random_walk_with_restart(g,query_index),query_index #sorted_scores


def load_topic_matrix(topic_id):
    #STEP 1:  LOAD GRAPH ##########################
    #> check that graph exists
    if not os.path.exists(get_sim_matrix_path(topic_id)):
        print (">> SIM MATRIX DOES NOT EXIST for: "+topic_id+": "+str(get_sim_matrix_path(topic_id)))
        run_pipeline()
    g,query_sentence,sims=load_sim_matrix_to_igraph(topic_id)
    query_node=g.vs.find(label=query_sentence)
    query_index=query_node.index
    ###############################################
    return g,query_sentence,sims,query_node,query_index

class NestedDict(dict):
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, NestedDict())


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_clustering_on_graph(topic_id='',method='fast_greedy',experiment=''):
    if not topic_id:topic_required=no_globals
    g,query_sentence,sims,query_node,query_index=load_topic_matrix(topic_id)
    #method='betweenness'   #talking to much time, we will skip it
    #method='walktrap'
    #method='leading_eigenvector'
    #g=filter_graph(g)
    print ("EXPERIMENT: "+str(experiment))
    
    if 'do7_sum_nodes' in experiment:
        #after doing the graph by the cos sim matrix, rank the sentences according to total score.
        clusters=[]
        cluster_weights=[]
        return g,clusters,cluster_weights,query_sentence,query_index

    elif 'do6' in experiment:
        print ("Calculate random walk scores...")
        g_random_walk_scores=calc_random_walk_with_restart(g, query_index)

        print ("Get edge distributions")
        query1_cosim_values=[]
        query2_cosim_values=[]
        cosim_values=[]
        ws_values=[]
        for i,e in enumerate(g.es): #FOR EACH EDGE
            #edge weight#  weight=e['weight']

            # Get nodes of edge
            vertex1=g.vs[e.tuple[0]] #node1 idx
            vertex2=g.vs[e.tuple[1]] #node2 idx
            
            # Get node walk scores
            walk_score1,vertex1,rank1=g_random_walk_scores[vertex1['s_idx']]
            walk_score2,vertex2,rank2=g_random_walk_scores[vertex2['s_idx']]
            
            #Get random walk score average
            rws_avg=(walk_score1+walk_score2)/2
            
            #Get edge cos sim score (current weight)
            edge_cosim=sims[vertex1['s_idx']][vertex2['s_idx']]
            
            query_cosim1=sims[query_index][vertex1['s_idx']] #For do6_2
            query_cosim2=sims[query_index][vertex2['s_idx']] #For do6_2
            
            #Store values for normalization
            query1_cosim_values+=[query_cosim1] #For do6_2
            query2_cosim_values+=[query_cosim2] #For do6_2
            cosim_values+=[edge_cosim]
            ws_values+=[rws_avg]
           
        #Calculate percent distributions
        if True: #Do rank based
            query1_cosim=calc_rank_percent_distribution(query1_cosim_values) #For do6_2
            query2_cosim=calc_rank_percent_distribution(query2_cosim_values) #For do6_2
            cosim_dist=calc_rank_percent_distribution(cosim_values)
            ws_dist=calc_rank_percent_distribution(ws_values)
        else:
            query1_cosim=normalize_max_min(query1_cosim_values) #For do6_2
            query2_cosim=normalize_max_min(query2_cosim_values) #For do6_2
            cosim_dist=normalize_max_min(cosim_values)
            ws_dist=normalize_max_min(ws_values)
        
        #Use percent distributions to calculate new weight
        for i,e in enumerate(g.es): #FOR EACH EDGE
            if experiment=='do6_two_scores_1':
                weight=cosim_dist[i]  #max of cosim OR ws_dist     (do6_1): edge_weight= ( Max[(cos sim) , [(node1_rws)+(node2_rws)]/2)] ]
            elif experiment=='do6_two_scores_2':
                weight=(cosim_dist[i]+(query1_cosim[i]+query2_cosim[i])/2)/2   #  (do6_2): edge_weight=((cos sim) + [(node1_qcs+node2_qcs)/2])/2
            else: #Standard do6_two_scores
                weight=(cosim_dist[i]+ws_dist[i])/2

#            print ("WEIGHT from: "+str(g.es[i]['weight'])+" to: "+str(weight)+" via: "+str(cosim_dist[i])+" and "+str(ws_dist[i]))
            g.es[i]['weight']=weight

    else:
        pass #Standard running no experiments
    
    
    communities=[]
    clusters=[]
    print ("Running clustering ["+method+"] on graph...")
    Perf.start()
    
    markov_subgraphs=[]
    uG='' #Undirected version of graph that corresponds to true cluster
    
    if 'markov' in method:
        print ("---> doing markov clustering")
        matrix=to_sparse(g,weight_attr='weight')
        result = mc.run_mcl(matrix)           # run MCL with default parameters
        clusters = mc.get_clusters(result) 
#D#        print ("GOT CLUSTERS: "+str(clusters))

        #Initialize subgrpah
        for idx,v in enumerate(g.vs):
            g.vs[idx]['subgraph']=''

        for cluster in clusters:
            cluster_id=cluster[0]
            indexes=cluster[1:]
            if not indexes:continue
            print ("markov cluster "+str(cluster_id)+" has: "+str(indexes))
            
            #Store subgraph index into back into igraph
            for idx in indexes:
                g.vs[idx]['subgraph']=cluster_id

        #Add subgraph cluster indexes back into graph
        print ("AT 0: "+str(g.vs[0]))
        #cl = VertexClustering(g, attribute='subgroup') #'g.vs["subgroup"])
        VC = VertexClustering(g)
        clusters = VC.FromAttribute(g, 'subgraph')
        #print ("GOT CLUSTER: "+str(cl))
    
    if 'betweenness' in method:
        #cluster_count=15
        print ("**betweenness requires edge trimming.  Doing that now...")
        g=filter_graph(g,the_type='remove_low_weights')
        print ("Calculating edge betweenness...")
        communities=g.community_edge_betweenness(clusters=cluster_count,weights='weight') #directed=
        print ("Fixing/checking dendogram -- must be fully connected.")
        communities=fix_dendrogram(g, communities)
    #########################################################

    if 'fast_greedy' in method:
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
    if experiment=='do3_avg_cosims':
        #The weight of the cluster will be the average of two values: the average cosine similarity (the
        # existing one in the code now) AND the average value of cosine similarity between all pairs (without 
        #the query).
        for i,subgraph in enumerate(clusters.subgraphs()):
            edge_sums=0

            #FOR EACH cluster

            #[1] QUERY TO EACH SENTENCE
            for idx, v in enumerate(subgraph.vs):
                #Sim between [QUERY_SENTENCE] and [Sentence at s_idx]
                edge_sim=sims[query_index][v['s_idx']] #Use stored sentence index to look up old cosine sim value
                edge_sums+=edge_sim
                
            #[2] SENTENCE to SENTENCE
            all_sum=0
            sum_count=0
            for idx1, v1 in enumerate(subgraph.vs):
                if idx1==query_index:continue     #skip query indexes
                for idx2, v2 in enumerate(subgraph.vs):
                    if idx2==query_index:continue #skip query indexes
                    if idx1==idx2: continue      #skip same sentences
                    all_sum+=sims[v1['s_idx']][v2['s_idx']]
                    sum_count+=1
                    
            ## Calc Cluster weights
            avg_weight=edge_sums/subgraph.vcount()
            if sum_count:
                avg_inter_weights=((all_sum)/2)/sum_count #2 cause double count
            else:
                avg_inter_weights=0
            print ("[experiment info]:  Cluster #"+str(i)+" inter weights average: "+str(avg_inter_weights))
            
            cluster_weights+=[(avg_weight+avg_inter_weights)/2]
    elif experiment=='do4_median_weight':
        #The weight of a cluster will be as:
        # Compute the average vector for all sentences in the cluster (each sentence is a vector, 
        #   compute the average vector to get the median vector) 
        # Then compute the cos sim between the query vector and the median vector.
        # This score is the weight of the cluster.

        pass
    else:
        for i,subgraph in enumerate(clusters.subgraphs()):
            edge_sums=0
            for idx, v in enumerate(subgraph.vs):
                if False:
                    print ("GOT: "+str(v.attribute_names()))
                    print ("Node: "+str(v['label']))
                edge_sim=sims[query_index][v['s_idx']] #Use stored sentence index to look up old cosine sim value
                edge_sums+=edge_sim
            avg_weight=edge_sums/subgraph.vcount()
            cluster_weights+=[avg_weight]
    #D        print ("Cluster #"+str(i)+" has node count: "+str(subgraph.vcount())+" avg weight: "+str(avg_weight))
                

    print ("For topic: "+str(topic_id)+" Done clustering took: "+str(time_clustering)+" seconds")
    
    if False:
        #Dview    output_clusters(g,communities,clusters,cluster_weights=cluster_weights)
        #Dview    g.write_pickle(fname="save_clustered_graph.dat")
        print ("Visualize clusters...")
        view_graph_clusters(g,clusters)
        
    return g,clusters,cluster_weights,query_sentence,query_index,uG


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def do_selection_by_weight(g,clusters,cluster_weights,query_sentence,query_index,top_j_clusters=3,target_sentences=5):
    #a)  Use only top j clusters

    ##/  Sort clusters by weight
    ptr_tuple=[]
    for idx,weight in enumerate(cluster_weights):
        ptr_tuple+=[(idx,weight)]
    ptr_tuple=sorted(ptr_tuple,key=lambda x:x[1],reverse=True)
    
    ##/  Random walk with restart calculation
    g_random_walk_scores=calc_random_walk_with_restart(g, query_index)

    ##3/ Lookup random walk scores for clusters
    #i_cluster:  The index of the sub-graph (cluster)
    #vc_index:   The index of a vertex in the sub-graph
    #s_idx:      The original index of the sentence/vertex in g
    rws_lookup={}
    cluster_sizes=[]
    for i_cluster,weight in ptr_tuple:
        subgraph=clusters.subgraphs()[i_cluster]
        for vc_index, v in enumerate(subgraph.vs):
            s_idx=v['s_idx']
            walk_score,vertex,rank=g_random_walk_scores[s_idx]
            #Store lookup between vertex index and walk scores
            if not i_cluster in rws_lookup: rws_lookup[i_cluster]=[]
            rws_lookup[i_cluster]+=[(vc_index,s_idx,walk_score)]
            
    

    ##/  Calc total weight of top j clusters
    target_sentences_per_cluster={}
    total_weights=0
    c=0
    for i_cluster,weight in ptr_tuple:
        total_weights+=weight
        c+=1
        if c==top_j_clusters:break
    ##/  Calc target sentences for each cluster
    c=0
    for i_cluster,weight in ptr_tuple:
        target_sentences_per_cluster[i_cluster]=int(round(weight/total_weights*target_sentences))
        c+=1
        if c==top_j_clusters:break
        
    
    ##4/  Random walk sort within each cluster
    print ()
    cache_sentences=[]
    cache_one_last_sentence_each=[] #If count<250 add 1 more sentence from EACH cluster
    token_count=0
    cc=0
    for i_cluster,weight in ptr_tuple:
        print ("---- Selection on cluster #"+str(i_cluster)+", weight:"+str(weight)+" ----------- choosing "+str(target_sentences_per_cluster[i_cluster])+" sentences.")
        subgraph=clusters.subgraphs()[i_cluster]
        
        #Sort walk scores for each cluster
        rws_sorted=sorted(rws_lookup[i_cluster], key=lambda x:x[2],reverse=True) #Sort by walk score
        
        ##/  Select top n
        #top_n=3
        #proportional_size=subgraph.vcount()/g.vcount()
        top_n=target_sentences_per_cluster[i_cluster]

        c=0
        for vc_index,s_idx,walk_score in rws_sorted:
            #Ignore original query
            if s_idx==query_index:continue  #Skip query_index
            c+=1
            vertex=subgraph.vs[vc_index]
            sentence=vertex['label']
            print ("Cluster #"+str(i_cluster)+" Top Score #"+str(c)+": %.9f"%walk_score+" >"+str(sentence))
            
            if c<=top_n:
                cache_sentences+=[sentence]
                token_count+=len(re.split(r' ',sentence))
            elif c==(top_n+1):
                cache_one_last_sentence_each+=[sentence] #If count<250 add 1 more sentence from EACH cluster
            else:
                break

        cc+=1
        if cc==top_j_clusters:break
        
    #Rounding can cause>target_sentences so limit
    cache_sentences=cache_sentences[:target_sentences]

    ##/ Special case:  If tokens<250 then add one more sentence from each cluster
    if token_count<250:
        print ("[special case] adding sentences as Summary < 250 tokens:")
        cache_sentences+=cache_one_last_sentence_each
        print(cache_one_last_sentence_each)
    
    return cache_sentences

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def do_selection(g,clusters,cluster_weights,query_sentence,query_index):
    ##
    #  Grab top sentences from each cluster
    ############################################
    weight_threshold=0.0078   
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
            sentence=vertex['label']
            print ("Cluster #"+str(i_cluster)+" Top Score #"+str(c)+": %.9f"%walk_score+" >"+str(sentence))
            yield  sentence
            if c==top_n:break
        
    print ("Done do_selection")   
    return

def alg_sort_clusters_by_weight(cluster_weights):
    #- rather then duplicating data structure (memory) adjust pointers to clusters
    ptr_tuple=[]
    for i,weight in enumerate(cluster_weights):
        ptr_tuple+=[(i,weight)]
    ptr_tuple=sorted(ptr_tuple,key=lambda x:x[1],reverse=True)
    return ptr_tuple

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def do_selection_by_round_robin(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=5,trim_low_weights=True,uG='',topic_id=''):
    global DOCS_SOURCE
    if str(DOCS_SOURCE)=='2006':
        print ("Removing weight threshold for 2006 dataset...")
        trim_low_weights=False

    ## Visual
    DO_VISUALIZATION=True
    if DO_VISUALIZATION:
        print ("Close visualization to continue...")
        if uG: #undirect graph
            _plot(topic_id,uG.copy(),membership=clusters)
        else: 
            _plot(topic_id,g.copy(),membership=clusters)
    

    ##:  Round robin selection:  Choose top sentence from each cluster in round-robin style

    ###  Standard info
    query_node=g.vs.find(label=query_sentence)
    query_index=query_node.index

    ##/  Random walk with restart calculation
    g_random_walk_scores=calc_random_walk_with_restart(g, query_index)
    

    ##1/  Sort clusters by weight
    ptr_tuple=alg_sort_clusters_by_weight(cluster_weights)


    ##2/  Filter clusters by weight threshold
    total_weight=0
    total_weight_count=0
    if trim_low_weights:
        weight_threshold=0.0095   
        ptr_tuple_top=[]
        for i,weight in ptr_tuple:
            total_weight+=weight
            total_weight_count+=1
            if weight>weight_threshold:
                ptr_tuple_top+=[(i,weight)]
    else:
        ptr_tuple_top=[]
        for i,weight in ptr_tuple:
            total_weight+=weight
            total_weight_count+=1
            ptr_tuple_top+=[(i,weight)]
    
    average_weight=total_weight/total_weight_count
    if not ptr_tuple_top:
        print ("ALL WEIGHTS BELOW THRESHOLD: "+str(weight_threshold))
        print ("Average weight: "+str(average_weight))
        print ("Suggest to modify weight threshold !")
        weight_=bad_threshold

    ##3/ Lookup random walk scores for clusters
    #        i_cluster:  The index of the sub-graph (cluster)
    #        vc_index:   The index of a vertex in the sub-graph
    #        s_idx:      The original index of the sentence/vertex in g
    rws_lookup={}
    for i_cluster,weight in ptr_tuple_top:
        subgraph=clusters.subgraphs()[i_cluster]
        for vc_index, v in enumerate(subgraph.vs):
            s_idx=v['s_idx']
            sentence=v['label']
            walk_score,vertex,rank=g_random_walk_scores[s_idx]
            #Store lookup between vertex index and walk scores
            if not i_cluster in rws_lookup: rws_lookup[i_cluster]=[]
            rws_lookup[i_cluster]+=[(vc_index,s_idx,walk_score,sentence)]
    

    ##4/  Random walk sort within each cluster
    print ()
    sorted_sentences_in_cluster={}
    cluster_ptr={}
    for i_cluster,weight in ptr_tuple_top: #FOR EACH CLUSTER
        #print ("---- Sort cluster sentences by walk score ----")
        cluster_ptr[i_cluster]=0        #Initialize sentence pointer in each cluster
        subgraph=clusters.subgraphs()[i_cluster]
        rws_sorted=sorted(rws_lookup[i_cluster], key=lambda x:x[2],reverse=True) #Sort by walk score
        
        #Save sorted sentences for round-robin lookup
        sorted_sentences_in_cluster[i_cluster]=rws_sorted
        

    ##5/  Round Robin Selection
    print ("Doing round robin selection")
    min_length=4
    max_sentence_length=90
    print ("0v8  require min sentence summary length of: "+str(min_length))
    print ("0v9  require max sentence word length summary length of: "+str(max_sentence_length))
    
    sentence_cache=[] #Top sentences to collect
    c=0
    lverbose=False
    while len(sentence_cache)<target_sentences: #while need more sentences
        c+=1
        
        ## CATCH IF CAN'T FIND ENOUGH GOOD SENTENCES
        if c==1000 or not ptr_tuple_top:
            print ("Could not resolve round-robin")
            print ("PTR: "+str(ptr_tuple_top))
            lverbose=True

            ## REVIEW ISSUE
            print ("-- DUMP cluster sentences")
            for i_cluster,weight in ptr_tuple_top: #FOR EACH CLUSTER
                print ("--------------------------------")
                rws_sorted=sorted_sentences_in_cluster[i_cluster] #Get sorted sentences for each cluster
                for sentence in rws_sorted:
                    print ("> "+str(sentence))
                    
            for i_cluster,weight in ptr_tuple_top: #FOR EACH CLUSTER
                rws_sorted=sorted_sentences_in_cluster[i_cluster] #Get sorted sentences for each cluster
                print ("[debug] POOL OF SENTENCES IN CLUSTER #"+str(i_cluster)+": "+str(len(rws_sorted)))
                print ("[debug] cluster pointers: "+str(cluster_ptr[i_cluster]))
            sys.exit("Could not resolve round-robin (see info above)")


        for i_cluster,weight in ptr_tuple_top: #FOR EACH CLUSTER
            rws_sorted=sorted_sentences_in_cluster[i_cluster] #Get sorted sentences for each cluster

            if cluster_ptr[i_cluster]<len(rws_sorted):                      #If pointer within length of array
                vc_index,s_idx,walk_score,sentence=rws_sorted[cluster_ptr[i_cluster]]

            if cluster_ptr[i_cluster]<len(rws_sorted):                      #If pointer within length of array
                got_next=True
                if s_idx==query_index: #exception when query index
                    got_next=False
                    cluster_ptr[i_cluster]+=1
                    if cluster_ptr[i_cluster]<len(rws_sorted):      #Case where only 1 sentence
                        vc_index,s_idx,walk_score,sentence=rws_sorted[cluster_ptr[i_cluster]]
                        got_next=True
                        
                #0v8# Require min length of 7
                if got_next and len(re.split(r'\s+',sentence))<min_length:
                    print ("#0v8# Skipping short sentence: "+str(sentence))
                    got_next=False
                    
                #0v9# Max length 90
                elif len(re.split(r' ',sentence))>max_sentence_length:
                    print ("#0v9# Skipping long sentence: "+str(sentence))
                    got_next=False

                elif got_next:
                    cluster_size=len(clusters.subgraphs()[i_cluster].vs)
                    print ("[RR] Storing sentence #"+str(len(sentence_cache)+1)+" from cluster: "+str(i_cluster)+"'s rank: "+str(cluster_ptr[i_cluster])+" cluster size: "+str(cluster_size)+">> "+str(sentence))
                    sentence_cache+=[sentence]
                    
                else:
                    print ("> skip non indexed: "+str(sentence))

                cluster_ptr[i_cluster]+=1 #Balanced increment
            if len(sentence_cache)==target_sentences:break
            
    print ("Done do_selection")   
    return sentence_cache


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def calc_random_walk_with_restart(g,query_index):
    #0v2# Add weight as a factor  g.es["weight"]
    
    #DEBUG LOOK AT GRAPH
    if False:
        c=0
        for v1,v2,weight in iter_graph_edges(g):
            c+=1
            print ("V1: "+v1['label'])
            print ("V2: "+v2['label'])
            print ("weight: "+str(weight))
            if c>10:break
        
    echo_top_scores=6
    if echo_top_scores:
        print ("---> Random walk top scores:")
    
    ## Calc random walk
    random_walk_with_restart=g.personalized_pagerank(reset_vertices=query_index,weights=g.es['weight'])
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
def select_top_cos_sims(topic_id='d301i',top_n=10,verbose=True):
    # Get sim matrix, sort and output top-n
    top_sentences=[]
    print ("SELECT TOP COS SIM FOR TOPIC: "+str(topic_id)+"--------")

    #STEP 1:  LOAD COS SIM MATRIX #################
    query_sentence=get_query(topic_id)
    print ("Query sentence: "+str(query_sentence))

    ## Get sentences
    documents,sentences,sentences_topics=files2sentences(limit_topic=topic_id) #watch loading 2x data into mem
    sentences.insert(0,query_sentence)
    
    ## Load sim matrix
    try:
        sims=load_sim_matrix(topic_id,zero_node2node=False) #Don't zero out diagonal to test scoring sort
    except:
        print ("NO SIM MATRIX FOR: "+topic_id)
        print ("Auto run pipeline to calc sim matrix")
        run_pipeline(use_specific_topic=topic_id)
        sims=load_sim_matrix(topic_id,zero_node2node=False) #Don't zero out diagonal to test scoring sort


    ## Sort cos sims
    query_sims=sims.tolist()[0] #First is query -- but set to 0 for itself.
    sorted_scores = sorted(zip(query_sims, range(len(query_sims))), key=lambda x: x[0],reverse=True) #G.vs is node id list
    
    ## output scores
    c=-1
    for cos_sim,idx in sorted_scores:
        c+=1
        if c>0:
            sentence=sentences[idx]
            top_sentences+=[sentence]
            if verbose:
                print ("#"+str(c)+" score: "+str(cos_sim)+" sentence idx: "+str(idx)+" sentence: "+str(sentence))
        if c==top_n: break

    return top_sentences


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def do1_select_query_cluster(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    #1)    Consider only the cluster that has the query sentences and ignore all other clusters, then sort 
    #its sentences according to the global random walk scores. Then choose top 10 sentences for the summary 
    #(without the query sentence).  
    
    ##/  Random walk with restart calculation
    g_random_walk_scores=calc_random_walk_with_restart(g, query_index)

    ##/ get cluster with query sentence
    query_sentence_in_clusterX=-1
    for cluster_idx,subgraph in enumerate(clusters.subgraphs()):
        clusterX_walk_scores=[]
        for vc_index, v in enumerate(subgraph.vs):
            s_idx=v['s_idx']
            sentence=v['label']
            
            # Remember walk score for sentence
            walk_score,vertex,rank=g_random_walk_scores[s_idx]
            clusterX_walk_scores+=[(s_idx,sentence,walk_score)]
                        
            if s_idx==query_index:
                query_sentence_in_clusterX=cluster_idx
        if query_sentence_in_clusterX>=0:
            break
    print ("Query sentence found in cluster #"+str(query_sentence_in_clusterX))

    ##/  Sort remembered sentence clusters
    clusterX_sentences_sorted_by_walk_score=sorted(clusterX_walk_scores, key=lambda x:x[2],reverse=True) #
    
    top_sentences=[]
    for s_idx,sentence,score in clusterX_sentences_sorted_by_walk_score:
        if s_idx==query_index:continue
        top_sentences+=[sentence]
        if len(top_sentences)==10:break
    return top_sentences

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def do2_local_walk(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=10,topic_id=''):
    #2)    After finish the clustering, do local random walk algorithm for each cluster subgraph (you should 
    #include the query node within each cluster and make it the reset vertex in the algorithm) then sort 
    #the sentences in each cluster according its own random walk score (ignore the global random walk scores here).
    
    ##1/  Sort clusters by weight
    ptr_tuple=alg_sort_clusters_by_weight(cluster_weights)

    ##/  For each cluster -- do random walk
    #    - get top sentences within cluster
    sorted_sentences_in_cluster={}
    active_clusters=[]
    cluster_ptr={}
    for cluster_idx,subgraph in enumerate(clusters.subgraphs()):
        cluster_ptr[cluster_idx]=0  #Initialize
        
        ## Filter clusters by weight threshold
        if cluster_weights[cluster_idx]<0.009:continue  #Filter low weight clusters
        active_clusters+=[cluster_idx]
        
        #If query not in subgraph then add
        try:
            query_node=subgraph.vs.find(s_idx=query_index) #
            idx_query_index=query_node.index
        except:
            query_node=''
        if not query_node: #Add query node to cluster sub-grpah
            subgraph,idx_query_index=add_vertex_to_graph(subgraph,s_idx=query_index,label=query_sentence,s_topic='')


        #Do random walk on sub-graph
        subg_random_walk_scores=calc_random_walk_with_restart(subgraph, idx_query_index)
                
        #Sort sentences within cluster by their local walk scores
        sorted_scores = sorted(zip(subg_random_walk_scores, subgraph.vs), key=lambda x: x[0],reverse=True) #G.vs is node id list
        
        sorted_sentences_in_cluster[cluster_idx]=sorted_scores
        
    #Output via Round Robin
    sentence_cache=[] #Top sentences to collect
    while len(sentence_cache)<target_sentences: #while need more sentences
        for i_cluster,weight in ptr_tuple: #FOR EACH CLUSTER
            if not i_cluster in active_clusters:continue #filtered

            sorted_scores=sorted_sentences_in_cluster[i_cluster] #Get sorted sentences for each cluster
            subg_walk_score,vertex=sorted_scores[cluster_ptr[i_cluster]]

            if cluster_ptr[i_cluster]<len(sorted_scores):                      #If pointer within length of array
                got_next=True
                if vertex['s_idx']==query_index: #exception when query index
                    got_next=False
                    cluster_ptr[i_cluster]+=1
                    if cluster_ptr[i_cluster]<len(sorted_scores):      #Case where only 1 sentence
                        subg_walk_score,vertex=sorted_scores[cluster_ptr[i_cluster]]
                        got_next=True

                if got_next:
                    print ("[RR] Storing sentence #"+str(len(sentence_cache)+1)+" from cluster: "+str(i_cluster)+"'s rank: "+str(cluster_ptr[i_cluster]))
                    sentence_cache+=[vertex['label']]

                    cluster_ptr[i_cluster]+=1
            if len(sentence_cache)==target_sentences:break
    return sentence_cache
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def do3_avg_cosims(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    #3)    The weight of the cluster will be the average of two values: the average cosine similarity (the
    # existing one in the code now) AND the average value of cosine similarity between all pairs (without 
    #the query).
    #To elaborate, consider the cos sim matrix for one of the clusters subgraph as:
    #querySentence 1Sentence 2Sentence 3Query1valuevaluevalueSentence 1value1valuevalueSentence
    # 2valuevalue1valueSentence 3valuevaluevalue1
    #So the first value as you did is the average of the red highlighted values.
    #Now, the second value will be the average of the yellow highlighted values.
    #Finally the cluster weight will be the average of these two values.
    target_sentences=10

    return do_selection_by_round_robin(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=target_sentences)


class Local_Vectorizer():
    def __init__(self,topic_id):
        self.load_vectorizers(topic_id)
        return

    def load_vectorizers(self,topic_id):
        tfidf_filename=TEMP_DATA_PATH+'tfidf_model_'+topic_id+'.mm'
        dictionary_filename=TEMP_DATA_PATH+'doc_dict'+topic_id+'.dict'
        
        self.dictionary = corpora.Dictionary.load(dictionary_filename)
        self.tfidf=models.TfidfModel.load(tfidf_filename)
        return
    
    def corpus2vector(self,corpus):
        norm_sentences=tokenize_sentences(corpus)
        raw_corpus = [self.dictionary.doc2bow(t) for t in norm_sentences]
        corpus_tfidf = self.tfidf[raw_corpus]
        return corpus_tfidf
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#4)    The weight of a cluster will be as:
#a.    Compute the average vector for all sentences in the cluster (each sentence is a vector, compute the average vector to get the median vector) 
#b.    Then compute the cos sim between the query vector and the median vector.
#c.    This score is the weight of the cluster.
def do4_median_weight(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    #Define:  average cluster vector == vector of all sentences in cluster
    print ("This requires tf-idf model for calculating cluster vectors")
    tfidf_filename=TEMP_DATA_PATH+'tfidf_model_'+topic_id+'.mm'
    if not os.path.exists(tfidf_filename):
        print ("Please run 'create_sim_matrix' / run_pipeline() to create the tfidf model")
        file_missing=hard_stop
    else:
        print ("Loading models...")
    
        
    #Calculate cluster vector based on sentences
    Vectorizer=Local_Vectorizer(topic_id)
    
    ## /  Get cluster sentences
    
    c_sentences=[]
    for subgraph in clusters.subgraphs():
        local_sentences=[]
        for vc_index, v in enumerate(subgraph.vs):
            s_idx=v['s_idx']
            sentence=v['label']
            local_sentences+=[sentence]
        #Add query sentence as first index before cluster corpus
        d_sentences=[query_sentence]
        d_sentences+=[" ".join(local_sentences)]

        c_sentences+=[d_sentences]
        
    new_cluster_weights=[]
    ## /  Calc average sentence vector
    for cluster_id, corpus in enumerate(c_sentences): #where corpus is query_sentence and cluster text
        #Calc average vector
        corpus_tfidf=Vectorizer.corpus2vector(corpus)

        #Calc similarity between query sentence and cluster text
        index = similarities.MatrixSimilarity(corpus_tfidf)
        #Look up similarities
        sims = index[corpus_tfidf]
    
        print ("Cluster #"+str(cluster_id)+" COSIM between query and cluster text vector: "+str(sims[0][1]))
        new_cluster_weights+=[sims[0][1]]
        if False:
            print ("  for sent1: "+str(corpus[0]))
            print ("   vs sent2: "+str(corpus[1]))

    return do_selection_by_round_robin(g,clusters,new_cluster_weights,query_sentence,query_index,target_sentences=10,trim_low_weights=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#5)    Try Markov clustering. 
def do5_markov_clustering(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    return do_selection_by_round_robin(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=10,trim_low_weights=True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#6)    
def do6_two_scores(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    #make the edge is between two nodes is actually (cos sim + (random walk score)) , 
    #then we insert this graph to the clustering, in other words, what if we add the random 
    #walk scores to the cos sim matrix, then make a graph from this new matrix that will be 
    #inserted to the clustering algorithim.
    
    #then (6) and yes if two edges can work in the clustering that will be good, but i am not sure
    #>> blend the percentiles score as the weight
    
    
    return do_selection_by_round_robin(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=12,trim_low_weights=True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def do6_two_scores_1(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    return do_selection_by_round_robin(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=12,trim_low_weights=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def do6_two_scores_2(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    return do_selection_by_round_robin(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=12,trim_low_weights=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def do7_sum_nodes(g,clusters,cluster_weights,query_sentence,query_index,topic_id=''):
    #after doing the graph by the cos sim matrix, rank the sentences according to total score.
    
    #Walk entire graph
    node_sums={}
    for i,e in enumerate(g.es): #FOR EACH EDGE
        weight=e['weight']

        # Get nodes of edge
        vertex1=g.vs[e.tuple[0]] #node1 idx
        vertex2=g.vs[e.tuple[1]] #node2 idx
        
        #Sum node on edge end of edge
        if not vertex1['s_idx']==query_index:
            try: node_sums[vertex1['s_idx']]+=weight
            except: node_sums[vertex1['s_idx']]=weight

        if not vertex2['s_idx']==query_index:
            try: node_sums[vertex2['s_idx']]+=weight
            except: node_sums[vertex2['s_idx']]=weight
        
    node_sums_sorted=sorted_x = sorted(node_sums.items(), key=operator.itemgetter(1),reverse=True) #by dict value
    print ("[debug] TOP SCORES: "+str(node_sums_sorted[:10]))
    
    sentences=[]
    c=0
    for s_idx,tot_weight in node_sums_sorted:
        c+=1
        sentences+=[g.vs[s_idx]['label']]
        if c==10:break
    return sentences

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__=='__main__':
    #branches=['select_top_cos_sims']
    branches=['run_clustering_on_graph']

    for b in branches:
        globals()[b]()





































































