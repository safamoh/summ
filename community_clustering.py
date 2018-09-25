from duc_reader import *

def load_sim_matrix_to_igraph():
    global TEMP_DATA_PATH,TOPIC_ID,LIMIT_TOPICS
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
    

    #STEP A:  Zero node-to-node simularity diagonal to 0def load_sim_matrix_to_igraph():
    global TEMP_DATA_PATH,TOPIC_ID,LIMIT_TOPICS
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

    return G,query_sentence

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
    method='fastgreedy'
    #method='betweenness'
    #method='walktrap'
    #method='spinglass'

    #STEP 1:  LOAD GRAPH ##########################
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

    print ("Done clustering took: "+str(time_clustering)+" seconds")
    
    if True:
        print ("Visualize clusters...")
        #view_graph_clusters(g,clusters)
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
