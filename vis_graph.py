import igraph
from random import randint

def _plot(g, membership=None):
    print ("Number of clusters: "+str(len(membership)))
    
    #Create lookup for clusters
    id2cluster={}
    for cluster_num,members in enumerate(membership):
        for idx in members:
            id2cluster[idx]=cluster_num

    if membership is not None:
        gcopy = g.copy()
        edges = []
        edges_colors = []
        for edge in g.es():
            #If not in same cluster then grey
            if id2cluster[edge.tuple[0]] != id2cluster[edge.tuple[1]]:
                edges.append(edge)
                edges_colors.append("gray")
            else:
                edges_colors.append("black")
        gcopy.delete_edges(edges)
        layout = gcopy.layout("kk")
        g.es["color"] = edges_colors
    else:
        layout = g.layout("kk")
        g.es["color"] = "gray"
    visual_style = {}
    visual_style["vertex_label_dist"] = 0
    visual_style["vertex_shape"] = "circle"
    visual_style["edge_color"] = g.es["color"]
    visual_style["vertex_size"] = 30
    # visual_style["bbox"] = (4000, 2500)
    visual_style["bbox"] = (1024, 768)
    visual_style["margin"] = 40
    
    #Standard 1)
    visual_style["layout"] = layout
    
    #Standard 2)
    #N??    visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=500, area=N ** 3, repulserad=N ** 3)
    #?need more    visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=1000)

    #Standard 3)  Supposed to be that circle thing
    #visual_style['layout']="fr" #

    if False:
        visual_style["edge_label"] = g.es["weight"] #noisy

    for vertex in g.vs():
        vertex["label"] = vertex.index

    if membership is not None:
        colors = []
        for i in range(0, len(membership)+1):
            colors.append('%06X' % randint(0, 0xFFFFFF))
        for vertex in g.vs():
#            vertex["color"] = str('#') + colors[membership[vertex.index]]
            vertex["color"] = str('#') + colors[id2cluster[vertex.index]]
        visual_style["vertex_color"] = g.vs["color"]

    print ("plotting...")
    igraph.plot(g, **visual_style)

if __name__ == "__main__":
    #fails# g = igraph.Nexus.get("karate")

    #cl = g.community_fastgreedy()
    #membership = cl.as_clustering().membership

    _plot(g, membership)