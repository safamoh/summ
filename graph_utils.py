from __future__ import division
import numpy as np
import random
from itertools import izip
from collections import defaultdict

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import TOPIC_ID
from duc_reader import LIMIT_TOPICS
from duc_reader import TEMP_DATA_PATH
from duc_reader import get_sim_matrix_path

import igraph
from igraph import plot  #pycairo  #pycairo-1.17.1-cp27-cp27m-win_amd64.whl https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def output_clusters(g,communities,clusters,cluster_weights=[]):
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
            print ("Cluster #"+str(i)+" has node count: "+str(subgraph.vcount())+" avg weight: "+str(cluster_weights[i]))
            for idx, v in enumerate(subgraph.vs):
                print ("Node: "+str(v['label']))
                if idx>3:break
    
        print ("Total number of clusters: "+str(len(clusters)))
    return

        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

def iter_graph_edges(g):
    for i,e in enumerate(g.es): #FOR EACH EDGE
        weight=e['weight']

        # Get nodes of edge
        vertex1=g.vs[e.tuple[0]] #node1 idx
        vertex2=g.vs[e.tuple[1]] #node2 idx
        yield vertex1,vertex2,weight
    return

def add_vertex_to_graph(g,**kwargs):
    g.add_vertices(1)
    idx=g.vcount()-1
    for kk in kwargs:
        g.vs[idx][kk]=kwargs[kk]
    return g,idx


def load_sim_matrix_to_igraph(local_topic_id=''):
    global TEMP_DATA_PATH,TOPIC_ID,LIMIT_TOPICS
    #LOAD STATE
    #####################################################
    
    if not local_topic_id:
        local_topic_id=TOPIC_ID
        local_limit_topics=LIMIT_TOPICS
    else:
        local_limit_topics=True

    query_sentence=get_query(local_topic_id)
    print ("Using query: "+str(query_sentence))

    if local_limit_topics:
        documents,sentences,sentences_topics=files2sentences(limit_topic=local_topic_id) #watch loading 2x data into mem
    else:
        documents,sentences,sentences_topics=files2sentences()
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,local_topic_id)
    #
    #############################


    #Reload simulation matrix
    print ("LOAD PRE-COMPUTED SIM MATRIX: "+str(get_sim_matrix_path(local_topic_id)))
    sims=np.load(get_sim_matrix_path(local_topic_id))
    

    #STEP A:  Zero node-to-node simularity diagonal to 0
    np.fill_diagonal(sims, 0)
    
    #STEP B:  Create iGraph
    G = igraph.Graph.Weighted_Adjacency(sims.tolist())
    G.vs['label'] = sentences   #node_names  # or a.index/a.columns
    
    #Add sentence index to graph so can look up in O(n) time.
    s_idx=range(len(sentences)) #Index to sentenes 0...
    G.vs['s_idx']=s_idx
    G.vs['s_topic']=sentences_topics #For by topic query
    
    #OUTPUT SAMPLE SIMS
    echo_sims(sims,sentences)

    return G,query_sentence,sims

def echo_sims(sims,sentences):
    # STEP 6:  Print sims 
    ###############################################
    options=[]
    options=['print_sims']
    print ("--> SAMPLE SIM NUMBERS (**note: node 2 node sims forced to 0 not 1.000):")

    if 'print_sims' in options:
        i=0
        j=0
        for item in list(enumerate(sims)):
            i+=1
            #            if i>0:break
            sent_num1=item[0]
            for sent_num2,cosim_value in enumerate(item[1]):
                j+=1
                idx="("+str(sent_num1)+","+str(sent_num2)+")"
                cosim_str="%.9f" % cosim_value
                if True and j<3:
                    print ("AT: "+str(idx)+" sim: "+str(cosim_str))
                    print ("  for sent1: "+str(sentences[sent_num1]))
                    print ("   vs sent2: "+str(sentences[sent_num2]))

#if same expect 0  -forced above                   if sentences[sent_num1]==sentences[sent_num2] and float(cosim_value)<0.9:
#if same expect 0                        print ("Low sim on similar sentences stop")
#if same expect 0                        a=hard_stop_check_logic

    return

def load_sim_matrix(local_topic_id,zero_node2node=True):
    #Reload simulation matrix
    sims=np.load(get_sim_matrix_path(local_topic_id))
    
    if zero_node2node:
        #STEP A:  Zero node-to-node simularity diagonal to 0
        np.fill_diagonal(sims, 0)
    return sims

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def rankVec(arg):
    p = np.unique(arg) #take unique value
    k = (p).argsort().argsort() #sort based on arguments in ascending order
    dd = defaultdict(int)
    for i in xrange(np.shape(p)[0]):
        dd[p[i]] = k[i]
    return np.array([dd[x] for x in arg])

def calc_rank_percent_distribution(vec):
    pranks=[]
    for rank in rankVec(vec): #1 is lowest
        rank=rank+1 #Start at 1 not 0
        pranks+=[rank/len(vec)]
#    print ("FIRST BUNCH RANK: "+str(vec[:10]))
#    print ("FIRST BUNCH RANK: "+str(pranks[:10]))
    return pranks
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def normalize_max_min(vec):
    the_max=max(vec)
    the_min=min(vec)
    pranks=[]
    for item in vec:
        if the_max==0:
            pranks+[0]
        else:
            pranks+=[(item-the_min)/the_max]
#    print ("MAX: "+str(the_max))
#    print ("MIN: "+str(the_min))
#
#    print ("FIRST BUNCH RANK: "+str(vec[:10]))
#    print ("FIRST BUNCH RANK: "+str(pranks[:10]))
    return pranks

def test_rank():
    vec=[3, 1, 4, 15, 92]
    print(rankVec(vec))
    return
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def markov_clustering():
    return

if __name__=='__main__':
    branches=['test_rank']
    branches=['markov_clustering']
    for b in branches:
        globals()[b]()







